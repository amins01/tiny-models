import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

TARGET_SIZE = 256 # 128
INPUT_DIR = "data/landscapes"
OUTPUT_DIR = f"data/landscapes_processed_masks_{TARGET_SIZE}"
OUTPUT_IMG_DIR = f"data/landscapes_processed_{TARGET_SIZE}"
BATCH_SIZE = 16

def create_lookup_table():
    mapping = {
        0: [2], # sky
        1: [4, 9, 17, 29, 35, 72, 94], # vegetation
        2: [16, 34, 42, 68], # mountain/rock
        3: [21, 26, 60, 104, 113, 128], # water
        4: [6, 11, 46, 52, 53, 54, 59, 91, 121], # road
        5: [13, 46, 91, 94] # sand
    }
    
    # fallback
    table = torch.full((151,), 6, dtype=torch.uint8) 

    for target_class, ade_indices in mapping.items():
        for ade_idx in ade_indices:
            if ade_idx < 151:
                table[ade_idx] = target_class
                
    return table

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        self.folder = folder
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        # Open and convert to RGB
        img = Image.open(path).convert("RGB")
        return img, self.files[idx]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    lookup_table = create_lookup_table().to(device)

    dataset = ImageDataset(INPUT_DIR)
    
    def collate_fn(batch):
        images = [b[0] for b in batch]
        names = [b[1] for b in batch]
        images = [img.resize((512, 512), Image.BILINEAR) for img in images]
        return images, names

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn)

    print("Starting extraction...")
    with torch.no_grad():
        for images, filenames in tqdm(loader):
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            logits_small = torch.nn.functional.interpolate(
                outputs.logits,
                size=(TARGET_SIZE, TARGET_SIZE),
                mode="bilinear",
                align_corners=False
            )
            
            pred_masks = logits_small.argmax(dim=1) # [bs, target, target]
            
            pred_masks = pred_masks.clamp(0, 150)
            simplified_masks = lookup_table[pred_masks]

            for i, filename in enumerate(filenames):
                mask_np = simplified_masks[i].cpu().numpy().astype(np.uint8)
                mask_path = os.path.join(OUTPUT_DIR, filename.replace(".jpg", ".png"))
                cv2.imwrite(mask_path, mask_np)
                
                img_small = images[i].resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BOX)
                img_small.save(os.path.join(OUTPUT_IMG_DIR, filename))

if __name__ == "__main__":
    main()