from data_loaders import LandscapeDataset, EditDataset

def get_dataset(name, args):
    if name == "landscapes":
        return LandscapeDataset(args.dataset_path, args.mask_path, args.image_size)

    if name == "omni_edit" or name == "gpt_edit":
        return EditDataset(args.dataset_path, args.metadata_path, args.image_size) # tasks={"style", "swap"}
    
    raise ValueError(f"Unsupported dataset {name}")