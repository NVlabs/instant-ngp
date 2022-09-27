import torchvision
import torch
import numpy as np

from PIL import Image
from pathlib import Path

KITTI_TEST_SEQS = unique_seqs = [
    '2011_09_26_drive_0002_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0013_sync',
    # '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0046_sync',
    # '2011_09_26_drive_0048_sync',
    # '2011_09_26_drive_0052_sync',
    # '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0086_sync',
    '2011_09_26_drive_0093_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0101_sync',
    '2011_09_26_drive_0106_sync',
    '2011_09_26_drive_0117_sync',
    '2011_09_28_drive_0002_sync',
    '2011_09_29_drive_0071_sync',
    '2011_09_30_drive_0016_sync',
    '2011_09_30_drive_0018_sync',
    '2011_09_30_drive_0027_sync',
    '2011_10_03_drive_0027_sync',
    '2011_10_03_drive_0047_sync',
]

class KITTI(torch.utils.data.Dataset):
    """KITTI dataset."""

    def __init__(
        self,
        path_gt: str,
        path_raw: str,
        split_type: str = 'eigen_with_gt',
        split: str = 'train',
        kb_crop: bool = False,
        kb_crop_gt: bool = False,
        sequence: str = None,
        normalize = False,
        path_nerf_renders: str = None,
    ):
        """
        Args:
            path_gt
            path_raw
            split
        """

        file_list = np.genfromtxt(Path(__file__).parent / 'splits' / split_type / f'{split}_files.txt', dtype=str)

        self.y_list = [
            Path(path_gt) / y
            for x, y in file_list if ((x.split('/')[1] == sequence) or (sequence is None))
        ]


        self.x_list = [
            Path(path_raw) / x
            for x, y in file_list if ((x.split('/')[1] == sequence) or (sequence is None))
        ]

        self.split = split
        self.kb_crop = kb_crop
        self.kb_crop_gt = kb_crop_gt

        self.path_nerf_renders = path_nerf_renders

        self.y_transforms = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
        ])

        if normalize:
            self.x_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.x_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])   


    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = Image.open(self.y_list[idx])
        x = Image.open(self.x_list[idx])

        if self.path_nerf_renders is not None:
            seq = str(self.x_list[idx]).split('/')[-4]
            x_nerf = Image.open(Path(self.path_nerf_renders) / seq / 'rgb' / self.x_list[idx].name)
            y_nerf = np.load(Path(self.path_nerf_renders) / seq / 'depth' / (self.x_list[idx].name[-15:-4] + '.npy'))

        y = self.y_transforms(y) / 256.0
        x = self.x_transforms(x)
        x_nerf = self.x_transforms(x_nerf)
        y_nerf = torch.from_numpy(y_nerf).unsqueeze(0)

        if self.kb_crop:
            _, H, W = x.shape 
            t = int(H - 352)
            l = int((W - 1216) / 2)
    
            x = x[:, t:t + 352, l:l+1216] 
            
            if self.kb_crop_gt:
                y = y[:, t:t + 352, l:l+1216] 

            print('WAAAAAAAAAAARNING!!!!')

        if self.path_nerf_renders is not None:
            return x, y, x_nerf, y_nerf
        return x, y
