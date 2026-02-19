import json
import torch
from torch.utils.data import Dataset
import cv2
import os
import albumentations as A

class LandscapeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=64, num_classes=8, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ], additional_targets={"mask": "mask"})

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        
        img_tensor = (torch.from_numpy(img).float() / 127.5) - 1.0 # normalize [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1) # [h, w, 3] -> [3, h, w]

        mask_tensor = torch.from_numpy(mask).long() # [h, w]
        mask_tensor = mask_tensor.clamp(0, self.num_classes - 1)
        mask_onehot = torch.nn.functional.one_hot(mask_tensor, num_classes=self.num_classes) # [h, w] -> [h, w, c]
        mask_onehot = mask_onehot.permute(2, 0, 1).float() # [c, h, w]

        return mask_onehot, img_tensor

class EditDataset(Dataset):
    def __init__(self, img_dir: str, metadata_path: str, image_size: int = 128, tasks: set[str] | None = None, augment: bool = True): # {"style", "env", "swap"}
        self.img_dir = img_dir
        self.metadata_path = metadata_path
        self.image_size = image_size
        self.augment = augment
        self.src_image_names = []
        self.tgt_image_names = []
        self.prompts = []
        self.tasks = []

        with open(self.metadata_path, mode="r", encoding="utf-8") as f:
            for l in f:
                json_l = json.loads(l)
                task = json_l.get("task")
                if not tasks or task in tasks:
                    self.src_image_names.append(json_l["src"])
                    self.tgt_image_names.append(json_l["tgt"])
                    self.prompts.append(json_l["txt"])
                    self.tasks.append(task)

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ], additional_targets={"tgt_img": "image"})

    def __len__(self):
        return len(self.src_image_names)

    def __getitem__(self, idx: int):
        src_img_name = self.src_image_names[idx]
        tgt_img_name = self.tgt_image_names[idx]
        prompt = self.prompts[idx]
        # task = self.tasks[idx]
        
        src_img_path = os.path.join(self.img_dir, src_img_name)
        tgt_img_path = os.path.join(self.img_dir, tgt_img_name)

        src_img = cv2.imread(src_img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        
        tgt_img = cv2.imread(tgt_img_path)
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        if self.augment:
            augmented = self.transform(image=src_img, tgt_img=tgt_img)
            src_img = augmented["image"]
            tgt_img = augmented["tgt_img"]

        src_img_tensor = (torch.from_numpy(src_img).float() / 127.5) - 1.0 # normalize [-1, 1]
        tgt_img_tensor = (torch.from_numpy(tgt_img).float() / 127.5) - 1.0 # normalize [-1, 1]

        src_img_tensor = src_img_tensor.permute(2, 0, 1) # [h, w, 3] -> [3, h, w]
        tgt_img_tensor = tgt_img_tensor.permute(2, 0, 1) # [h, w, 3] -> [3, h, w]

        return src_img_tensor, tgt_img_tensor, prompt
