import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import BaseModule
import einops
from torch import einsum

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


### Position ###
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def relative_logits_1d(q, rel_k):
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = expand_dim(logits, dim = 5, k = logits.shape[4])
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        kernel_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(kernel_size)
        scale = dim_head ** -0.5
        self.kernel_size = kernel_size
        self.dim_head = dim_head
        self.rel_height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        'q: b out_channels group_channels H W'
        b, out_channels, group_channels, h, w = q.shape

        q = q.permute(0, 1, 3, 4, 2)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.permute(0, 1, 5, 4, 2, 3).contiguous().view(b, out_channels, -1, h, w)

        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.permute(0, 1, 4, 5, 2, 3).contiguous().view(b, out_channels, -1, h, w)
        return rel_logits_w + rel_logits_h

### Position ###


class AT_kernel(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 group_channels=4,
                 reduction_ratio=4):
        super(AT_kernel, self).__init__()

        self.stride = stride
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

        self.kernel_size = kernel_size


        self.out_channels = out_channels
        self.group_channels = group_channels    # defult: 4

        self.emb_K = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )


        MLP_inchannel = in_channels * 2
        MLP_midchannel = in_channels // reduction_ratio
        self.MLP = nn.Sequential(
            nn.Conv2d(in_channels=MLP_inchannel, out_channels=MLP_midchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(MLP_midchannel),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=MLP_midchannel, out_channels=out_channels * group_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels * group_channels)
        )


        self.unfold_local = nn.Unfold(kernel_size=(kernel_size,kernel_size),dilation=1,padding=(kernel_size-1)//2,stride=1)

        
        self.emb_V = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * group_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * group_channels),
        )


    def forward(self, x):

        x = x if self.stride == 1 else self.avgpool(x)
        B, C, H, W = x.shape

        Q = einops.repeat(x, "b c h w -> b (c k) h w", k=self.kernel_size**2).view(B, C, self.kernel_size**2, H, W)
        K = self.unfold_local(self.emb_K(x)).view(B, C, self.kernel_size**2, H, W)   # [B C K×K H W]

        QK_concat = torch.cat((Q, K), dim=1).view(B, 2*C, (self.kernel_size**2)*H, W)
        attn = self.MLP(QK_concat).view(B, self.out_channels, self.group_channels, self.kernel_size**2, H, W)

        'attn: [B out_channel group_channel K×K H W]'


        V = self.unfold_local(self.emb_V(x)).view(B, self.out_channels, self.group_channels, self.kernel_size**2, H, W) # [B out_channel group_channel K×K H W]

    
        out = attn * V
        out = out.sum(dim=3).sum(dim=2).contiguous().view(B, self.out_channels, H, W)

        return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ATkernel=False, ATkernel_size=3, softmax=False, group_channels=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if ATkernel:
            self.conv2 = AT_kernel(in_channels=width, out_channels=width, kernel_size=ATkernel_size, stride=stride, softmax=softmax, group_channels=group_channels)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.ATkernel = ATkernel

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




@BACKBONES.register_module()
class ResNet_ATkernel(BaseModule):
    def __init__(self, layers, block=Bottleneck, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, 
                 replace_conv2_with_ATkernel=[1, 1, 1, 1], 
                 stride2or1block=[0, 0], 
                 resolution=224, 
                 ATkernel_size=3, 
                 group_channels=4,
                 init_cfg=None):
        super(ResNet_ATkernel, self).__init__(init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if resolution==32 or resolution==64:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
            self.layer1 = self._make_layer(block, 64, layers[0],
                                        ATkernel=replace_conv2_with_ATkernel[0], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        ATkernel=replace_conv2_with_ATkernel[1], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        ATkernel=replace_conv2_with_ATkernel[2], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        ATkernel=replace_conv2_with_ATkernel[3], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
        elif resolution==224:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0],
                                        ATkernel=replace_conv2_with_ATkernel[0], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        ATkernel=replace_conv2_with_ATkernel[1], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        ATkernel=replace_conv2_with_ATkernel[2], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
                                        
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        ATkernel=replace_conv2_with_ATkernel[3], stride2or1block=stride2or1block, ATkernel_size=ATkernel_size, group_channels=group_channels)
       

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * eval(block).expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, ATkernel=False, stride2or1block=[0, 0], ATkernel_size=3, group_channels=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, ATkernel=(ATkernel and stride2or1block[0]), ATkernel_size=ATkernel_size, group_channels=group_channels))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, ATkernel=(ATkernel and stride2or1block[1]), ATkernel_size=ATkernel_size, group_channels=group_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


if __name__ == "__main__":
    pass