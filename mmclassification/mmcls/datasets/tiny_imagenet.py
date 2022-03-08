import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class Tiny_ImageNet(BaseDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'tench, Tinca tinca',
        'goldfish, Carassius auratus',
        'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',  # noqa: E501
        'tiger shark, Galeocerdo cuvieri',
        'hammerhead, hammerhead shark',
        'electric ray, crampfish, numbfish, torpedo',
        'stingray',
        'cock',
        'hen',
        'ostrich, Struthio camelus',
        'brambling, Fringilla montifringilla',
        'goldfinch, Carduelis carduelis',
        'house finch, linnet, Carpodacus mexicanus',
        'junco, snowbird',
        'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
        'robin, American robin, Turdus migratorius',
        'bulbul',
        'jay',
        'magpie',
        'chickadee',
        'water ouzel, dipper',
        'kite',
        'bald eagle, American eagle, Haliaeetus leucocephalus',
        'vulture',
        'great grey owl, great gray owl, Strix nebulosa',
        'European fire salamander, Salamandra salamandra',
        'common newt, Triturus vulgaris',
        'eft',
        'spotted salamander, Ambystoma maculatum',
        'axolotl, mud puppy, Ambystoma mexicanum',
        'bullfrog, Rana catesbeiana',
        'tree frog, tree-frog',
        'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
        'loggerhead, loggerhead turtle, Caretta caretta',
        'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',  # noqa: E501
        'mud turtle',
        'terrapin',
        'box turtle, box tortoise',
        'banded gecko',
        'common iguana, iguana, Iguana iguana',
        'American chameleon, anole, Anolis carolinensis',
        'whiptail, whiptail lizard',
        'agama',
        'frilled lizard, Chlamydosaurus kingi',
        'alligator lizard',
        'Gila monster, Heloderma suspectum',
        'green lizard, Lacerta viridis',
        'African chameleon, Chamaeleo chamaeleon',
        'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',  # noqa: E501
        'African crocodile, Nile crocodile, Crocodylus niloticus',
        'American alligator, Alligator mississipiensis',
        'triceratops',
        'thunder snake, worm snake, Carphophis amoenus',
        'ringneck snake, ring-necked snake, ring snake',
        'hognose snake, puff adder, sand viper',
        'green snake, grass snake',
        'king snake, kingsnake',
        'garter snake, grass snake',
        'water snake',
        'vine snake',
        'night snake, Hypsiglena torquata',
        'boa constrictor, Constrictor constrictor',
        'rock python, rock snake, Python sebae',
        'Indian cobra, Naja naja',
        'green mamba',
        'sea snake',
        'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
        'diamondback, diamondback rattlesnake, Crotalus adamanteus',
        'sidewinder, horned rattlesnake, Crotalus cerastes',
        'trilobite',
        'harvestman, daddy longlegs, Phalangium opilio',
        'scorpion',
        'black and gold garden spider, Argiope aurantia',
        'barn spider, Araneus cavaticus',
        'garden spider, Aranea diademata',
        'black widow, Latrodectus mactans',
        'tarantula',
        'wolf spider, hunting spider',
        'tick',
        'centipede',
        'black grouse',
        'ptarmigan',
        'ruffed grouse, partridge, Bonasa umbellus',
        'prairie chicken, prairie grouse, prairie fowl',
        'peacock',
        'quail',
        'partridge',
        'African grey, African gray, Psittacus erithacus',
        'macaw',
        'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
        'lorikeet',
        'coucal',
        'bee eater',
        'hornbill',
        'hummingbird',
        'jacamar',
        'toucan',
        'drake',
        'red-breasted merganser, Mergus serrator',
        'goose',
        'black swan, Cygnus atratus',
        'tusker',
        'echidna, spiny anteater, anteater',
        'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',  # noqa: E501
        'wallaby, brush kangaroo',
        'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',  # noqa: E501
        'wombat',
        'jellyfish',
        'sea anemone, anemone',
        'brain coral',
        'flatworm, platyhelminth',
        'nematode, nematode worm, roundworm',
        'conch',
        'snail',
        'slug',
        'sea slug, nudibranch',
        'chiton, coat-of-mail shell, sea cradle, polyplacophore',
        'chambered nautilus, pearly nautilus, nautilus',
        'Dungeness crab, Cancer magister',
        'rock crab, Cancer irroratus',
        'fiddler crab',
        'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',  # noqa: E501
        'American lobster, Northern lobster, Maine lobster, Homarus americanus',  # noqa: E501
        'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',  # noqa: E501
        'crayfish, crawfish, crawdad, crawdaddy',
        'hermit crab',
        'isopod',
        'white stork, Ciconia ciconia',
        'black stork, Ciconia nigra',
        'spoonbill',
        'flamingo',
        'little blue heron, Egretta caerulea',
        'American egret, great white heron, Egretta albus',
        'bittern',
        'crane',
        'limpkin, Aramus pictus',
        'European gallinule, Porphyrio porphyrio',
        'American coot, marsh hen, mud hen, water hen, Fulica americana',
        'bustard',
        'ruddy turnstone, Arenaria interpres',
        'red-backed sandpiper, dunlin, Erolia alpina',
        'redshank, Tringa totanus',
        'dowitcher',
        'oystercatcher, oyster catcher',
        'pelican',
        'king penguin, Aptenodytes patagonica',
        'albatross, mollymawk',
        'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',  # noqa: E501
        'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
        'dugong, Dugong dugon',
        'sea lion',
        'Chihuahua',
        'Japanese spaniel',
        'Maltese dog, Maltese terrier, Maltese',
        'Pekinese, Pekingese, Peke',
        'Shih-Tzu',
        'Blenheim spaniel',
        'papillon',
        'toy terrier',
        'Rhodesian ridgeback',
        'Afghan hound, Afghan',
        'basset, basset hound',
        'beagle',
        'bloodhound, sleuthhound',
        'bluetick',
        'black-and-tan coonhound',
        'Walker hound, Walker foxhound',
        'English foxhound',
        'redbone',
        'borzoi, Russian wolfhound',
        'Irish wolfhound',
        'Italian greyhound',
        'whippet',
        'Ibizan hound, Ibizan Podenco',
        'Norwegian elkhound, elkhound',
        'otterhound, otter hound',
        'Saluki, gazelle hound',
        'Scottish deerhound, deerhound',
        'Weimaraner',
        'Staffordshire bullterrier, Staffordshire bull terrier',
        'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',  # noqa: E501
        'Bedlington terrier',
        'Border terrier',
        'Kerry blue terrier',
        'Irish terrier',
        'Norfolk terrier',
        'Norwich terrier',
        'Yorkshire terrier',
        'wire-haired fox terrier',
        'Lakeland terrier',
        'Sealyham terrier, Sealyham',
        'Airedale, Airedale terrier',
        'cairn, cairn terrier',
        'Australian terrier',
        'Dandie Dinmont, Dandie Dinmont terrier',
        'Boston bull, Boston terrier',
        'miniature schnauzer',
        'giant schnauzer',
        'standard schnauzer',
        'Scotch terrier, Scottish terrier, Scottie',
    ]

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
