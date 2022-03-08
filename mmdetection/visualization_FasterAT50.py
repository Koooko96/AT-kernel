from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import torch

config_file = 'configs/faster_rcnn_r50_ATv821_0001_11_fpn_1x_b4x2_coco.py'
checkpoint_file = 'To Be Fixed (path to the checkpoint)'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# init a detector
model = init_detector(config_file, checkpoint_file, device=device)

val_data_path = "To Be Fixed (.../Dataset/coco/val2017)"
images = os.listdir(val_data_path)
images.sort()

visualization_save_path = "COCO_visualization/FasterAT50"
if not os.path.exists:
    os.makedirs(visualization_save_path)

number = 100

for i in range(0, number):
    image_path = os.path.join(val_data_path, images[i])
    result = inference_detector(model, image_path)
    show_result_pyplot(model, image_path, result, out_file=os.path.join(visualization_save_path, images[i]))
