from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from .. import config

class MapDataset(Dataset):
    def __init__(self, root_dir: str) -> Dataset:
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    
    def __len__(self) -> int:
        return len(self.list_files)
    
    def __getitem__(self, index) -> np.ndarray:
        # read the image
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))

        # get width (h,w,c)
        w = image.shape[1]
        assert (w%2) == 0, f"width of image should be multple of 2, current image width: {w}"
        input_image = image[:, :w//2, :]
        target_image = image[:,w//2:, :]

        # apply transform
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations['image']
        output_image = augmentations['image0']

        input_image = config.transform_only_input(image=input_image)["image"]
        output_image = config.transform_only_mask(image=output_image)["image"]

        return input_image, output_image
