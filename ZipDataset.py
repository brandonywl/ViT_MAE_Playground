import cv2
import random
import zipfile

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, ToTensor
import torchstain

class ZipDataset(Dataset):
    def __init__(self, root_path, target_folders, classnames, transform, stain_normalize, get_mask=False):
        self.zip_file = zipfile.ZipFile(root_path, 'r')
        # self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        directories = self.zip_file.namelist()
        interested_directories = list(filter(lambda x: target_folders in x, directories))
        interested_files = list(filter(lambda x: x[-4:] == '.jpg', interested_directories))

        if get_mask:
            masked_files = list(filter(lambda x: "mask" in x, interested_files))
            image_files = [file.replace("mask", "data") for file in masked_files]
            self.label_list = extractClassName(classnames, image_files) if classnames is not None else None
            self.name_list = list(zip(image_files, masked_files))

        else:
            self.name_list = list(interested_files)
            self.label_list = extractClassName(classnames, self.name_list) if classnames is not None else None

        # self.label_list = extractClassName(classnames, self.name_list) if classnames is not None else None
        self.to_tensor = ToTensor()
        self.transforms = transform
        self.stain_normalize = stain_normalize
        self.get_mask=get_mask

    def __getitem__(self, key, SEED=0):
        name = self.name_list[key]
        if self.get_mask:
            name, mask_name = name

        buf = self.zip_file.read(name=name)
        img = self.to_tensor(cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR))

        if self.get_mask:
            buf = self.zip_file.read(name=mask_name)
            mask_img = self.to_tensor(cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR))
            # Need to transform mask also

        random.seed(SEED) # apply this seed to img transforms
        if self.transforms:
            img = self.transforms(img)
        if self.stain_normalize:
            normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            norm, H, E = normalizer.normalize(I=img, stains=True)
            img = norm.permute(2,0,1)

        return_package = (img,)

        if self.get_mask:
            return_package = return_package + (mask_img,)

        if self.label_list:
            label = self.label_list[key]
            return_package = return_package + (label,)
            # return img, label
        # else:
        #     return img

        return return_package

    def __len__(self):
        return len(self.name_list)
    
# Extracting the classname of the img from the file path
def extractClassName(classnames, filepaths):
    class_names = []
    for filepath in filepaths:
        for idx, class_name in enumerate(classnames):
            if class_name in filepath:
                class_names.append(idx)
                break
    if (len(class_names) == len(filepaths)):
        return class_names
    else:
        print("[ERROR] No matching record is found for at least one of the data sample.")



def get_default_transforms(desired_input_image_shape):
    default_img_transforms = v2.Compose([
        # Random Resizing Crop
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.Resize(desired_input_image_shape, antialias=True),

        # Horizontal Flipping Transformation
        v2.RandomHorizontalFlip(p=1),

        # Vertical Flipping Transformation
        v2.RandomVerticalFlip(p=1),

        # Color Transformation
        # v2.ColorJitter(brightness=(0.5,1.0), contrast=1,saturation=0, hue=0.4),

        v2.ToDtype(torch.float32),
    ])

    return default_img_transforms

def create_dataloader(data_path, target_folder, classnames, batch_size, input_image_shape, stain_normalize=False, get_mask=False):
    transform = get_default_transforms(input_image_shape)

    dataset = ZipDataset(data_path, target_folder, classnames, transform=transform, stain_normalize=stain_normalize, get_mask=get_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    return dataloader