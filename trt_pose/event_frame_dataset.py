import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import PIL.Image
import os

class EventFrameDataset(Dataset):
    """
    Dataset class for loading event frames.
    this dataset contains event frames and their corresponding rgb images. We also save the paf and cmap from rgb images and use them as gt labels to train a model for eventframe
    parameters:
        path (str): Path to the dataset.
    """
    def __init__(self, path, if_data_augmentation=True):
        self.if_data_augmentation = if_data_augmentation
        self.path = path
        self.rgb_path = os.path.join(path, 'rgb')
        self.event_frame_path = os.path.join(path, 'event_frame')
        self.paf_path = os.path.join(path, 'paf')
        self.cmap_path = os.path.join(path, 'cmap')
        filename_list = sorted(os.listdir(self.rgb_path))
        self.name_list = [os.basename(filename).replace('.jpg','') for filename in self.filename_list]
        self.length = len(self.name_list)
        if self.length == 0:
            raise ValueError("No files found in the specified directory.")
    
    def __normalize(self, rgb_tensor):
        """
        normalize to match the 
        """
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(rgb_tensor)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds.")
        rgb_name = self.name_list[idx]
        rgb_path = os.path.join(self.rgb_path, rgb_name + '.jpg')
        event_frame_path = os.path.join(self.event_frame_path, rgb_name + '.jpg')
        paf_path = os.path.join(self.paf_path, rgb_name + '.paf')
        cmap_path = os.path.join(self.cmap_path, rgb_name + '.cmap')

        rgb_image_pil = PIL.Image.open(rgb_path).convert('RGB')
        event_frame_pil = PIL.Image.open(event_frame_path).convert('L')

        rgb_gray_tensor = transforms.ToTensor()(rgb_image_pil.convert('L'))
        # Normalize the RGB tensor
        rgb_gray_tensor = self.__normalize(rgb_gray_tensor)
        event_frame_tensor = transforms.ToTensor()(event_frame_pil)

        if not os.path.exists(paf_path) or not os.path.exists(cmap_path):
            raise FileNotFoundError(f"Required files not found for {rgb_name}. Ensure paf and cmap files exist.")
        paf = torch.load(paf_path)
        cmap = torch.load(cmap_path)

        if self.if_data_augmentation:
            # 随机水平翻转
            if torch.rand(1).item() > 0.5:
                rgb_tensor = F.hflip(rgb_tensor)
                event_frame_tensor = F.hflip(event_frame_tensor)
                paf = F.hflip(paf)
                cmap = F.hflip(cmap)

            # 随机垂直翻转
            if torch.rand(1).item() > 0.5:
                rgb_tensor = F.vflip(rgb_tensor)
                event_frame_tensor = F.vflip(event_frame_tensor)
                paf = F.vflip(paf)
                cmap = F.vflip(cmap)

            # 随机旋转
            angle = torch.randint(-10, 10, (1,)).item()
            rgb_tensor = F.rotate(rgb_tensor, angle)
            event_frame_tensor = F.rotate(event_frame_tensor, angle)
            paf = F.rotate(paf, angle)
            cmap = F.rotate(cmap, angle)            

            # event frame 不分+-势能


        return {
            'rgb': rgb_gray_tensor,
            'event_frame': event_frame_tensor,
            'paf': paf,
            'cmap': cmap,
            'name': rgb_name
        }
        
    def get_part_type_counts(self):
        return torch.sum(self.counts, dim=0)
    
    def get_paf_type_counts(self):
        c = torch.sum(self.connections[:, :, 0, :] >= 0, dim=-1) # sum over parts
        c = torch.sum(c, dim=0) # sum over batch
        return c




