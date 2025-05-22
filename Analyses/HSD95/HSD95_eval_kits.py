"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (see LICENSE file or http://www.apache.org/licenses/LICENSE-2.0)
# This file includes components adapted from MONAI (Apache 2.0).
"""
# For the Kits dataset, this code iterates over 5 cross-validation folds created for each model type.
# It loads the corresponding model for each fold and evaluates it on that fold.
# The final results for HSD (Hausdorff Distance) are reported as the average over the 5 folds,
# along with the standard deviation.

import os
import tempfile
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from monai.config import print_config
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, CropForegroundd,
    Orientationd, Spacingd, EnsureTyped, AsDiscrete, KeepLargestConnectedComponent
)
from monai.data import (
    CacheDataset, load_decathlon_datalist, ThreadDataLoader
)
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from monai.networks.nets import UNETR, BasicUNet 

from BUNet.LocBAM.LocBAM_1D import BasicUnetLocBAMs2

from UNETR_alternative_PE import UNETRCoordEmbed

if __name__ == "__main__":
    mp.set_start_method("spawn")
    print_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4 #kits
    model_base_path = "/u/home/hodo/Documents/locunet/runs/KITS/KITS23_UNet_final_Baseline_small_0" #"/u/home/hodo/Documents/locunet/runs/AMOS/UNet_final_Baseline_small_"
#    0/0.pth  # Change as needed
    model_name = "KITS23_UNet_final_Baseline_small_"#"UNet_final_Baseline_small_"  # e.g., Final_UNETR_small_0.pth
    folds = 5  # K-Fold CV

    # Transforms for AMOS validation
    def get_val_transforms():
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            ScaleIntensityRanged(keys=["image"], a_min=-58, a_max=302, b_min=0.0, b_max=1.0, clip=True),
       
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1]),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        ])

    def build_dataset(patch_size, batch_size, crop_batch_size, fold_idx, max_samples):
        data_dir = "/vol/ciamspace/datasets/KiTs23/dataset/"
        split_json = "random_samples.json"
        datasets = os.path.join(data_dir, split_json)
        val_files = load_decathlon_datalist(datasets, True, "validation")
        val_files = val_files[:max_samples]  
        val_ds = CacheDataset(
            data=val_files,
            transform=get_val_transforms(),
            cache_rate=0.333,
            num_workers=4
        )
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        return val_loader

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    largest_connected = KeepLargestConnectedComponent(num_components=1)
    # patch_size = (128, 128, 128)
    # batch_size = 2
    patch_size= (32, 32, 32)
    batch_size = 128

    hd95_scores = []

    for fold_idx in range(folds):
        print(f"\n=== Evaluating Fold {fold_idx} ===")

        model_path = f"{model_base_path}{fold_idx}/{model_name}{fold_idx}.pth"

        # model = UNETR(
        #     in_channels=1,
        #     out_channels=num_classes,
        #     img_size=patch_size
        # ).to(device)

        # model = UNETRCoordEmbed(in_channels=1, out_channels=num_classes, img_size=img_size) #.to("cuda" if torch.cuda.is_available() else "cpu")

        model = BasicUNet(spatial_dims=3, 
                        in_channels=1, 
                        out_channels=num_classes, 
                        features=(32, 32, 64, 128, 256, 32)).to(device)

        # model = BasicUnetLocBAMs2(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=num_classes,
        #     features=(32, 32, 64, 128, 256, 32),
        #     dropout=0.1,

        #     hanet_params=None,
        # )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
        model.eval()

        val_loader = build_dataset(patch_size, batch_size, 1, fold_idx, max_samples=10)
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Validation"):
                val_inputs = batch["image"].to(device)
                val_labels = batch["label"].to(device)
                # Apply preprocessing depending on sampling method
                # if sampling == "Baseline":
                val_inputs = val_inputs[:, 0:1, :, :, :]
                
                model = model.half()
                val_inputs = val_inputs.half()

                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model)

                val_labels_list = [post_label(lbl) for lbl in batch["label"]]
                val_outputs_list = [post_pred(pred) for pred in val_outputs]
                val_outputs_list = [largest_connected(pred) for pred in val_outputs_list]

                hd95_metric(y_pred=val_outputs_list, y=val_labels_list)
                torch.cuda.empty_cache()  

        hd95_score = hd95_metric.aggregate().item()
        hd95_scores.append(hd95_score)
        print(f"HD95 (Fold {fold_idx}): {hd95_score:.4f}")
        hd95_metric.reset()

    print("\n=== Summary ===")
    print(f"Mean HD95: {np.mean(hd95_scores):.4f}")
    print(f"Std Dev HD95: {np.std(hd95_scores):.4f}")
