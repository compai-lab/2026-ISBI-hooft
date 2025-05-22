"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)
# This script uses MONAI (Apache 2.0) and PyTorch (BSD-style license) for medical image processing and deep learning.

Description:
This script performs cross-validated model evaluation for 3D medical image segmentation using MONAI and PyTorch.

Key functionality:
- Loads pretrained models (e.g., BasicUNet) across 5 cross-validation folds.
- Applies MONAI-based preprocessing pipelines for volumetric data (e.g., BTCV).
- Uses sliding window inference to evaluate each model on its corresponding validation set.
- Computes both Dice score and Surface Dice metrics per class and per fold.
- Optionally applies connected component filtering to retain the largest predicted structure.
- Aggregates and saves per-fold Dice results as JSON for further statistical analysis (e.g., Wilcoxon tests).

Notes:
- The model and dataset paths are hardcoded for BTCV; adapt them for other datasets if needed.
- Requires all fold checkpoints to be present for full evaluation.
"""



import os
import shutil
import tempfile
import json


import matplotlib.pyplot as plt
import monai.losses
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
ToTensord,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandCropByLabelClassesd,
    FgBgToIndicesd,
    ClassesToIndicesd,
    RandRotated,
SpatialPadd,
AddCoordinateChannelsd
)

from monai.config import print_config
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.networks.nets import SwinUNETR, FlexibleUNet, UNETR, ViTAutoEnc, BasicUNet
#added code ______top
from monai.data import ITKReader, NibabelReader
from monai.transforms import LoadImaged
#added code ________ bottom
from monai.data import (
DataLoader,
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
    pad_list_data_collate,
    SmartCacheDataset,
CacheNTransDataset,
ThreadBuffer
)
from BUNet.LocBAM.LocBAM_1D import BasicUnetLocBAMs2

import torch.multiprocessing as mp
import torch
#torch.multiprocessing.set_start_method("spawn")
if __name__ == "__main__":
    mp.set_start_method("spawn")
    print_config()
    # the train dataset and the cross-validation split configs are here
    # os.environ["MONAI_DATA_DIRECTORY"] = "/home/compai/code/data/nnunet_raw/nnunet_dataset/Dataset101_BTCV_abdomen/data/btcv/"
    
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)


    num_classes = 14 #14 for btcv dataset, 2 for lung_dataset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #

    def get_val_transforms():
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                

                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                #add chanmel for correct scores here 
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1]),
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1,2]),

                # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                
                EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            ]
        )
        return val_transforms


    def build_dataset(current_patch_size, sample_batch_size, crop_batch_size, fold_idx):
        data_dir = "/vol/ciamspace/datasets/btcv/" #or btcv)backup for w CurrREg scores
        # data_dir = "/vol/ciamspace/datasets/Dataset006_Lung/"
        split_json = "dataset_" + str(fold_idx) + ".json"

        datasets = data_dir + split_json
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        cache_rate = 1.0
        # train_ds = CacheDataset(
        #     data=datalist,
        #     transform=get_train_transforms(current_patch_size, crop_batch_size),
        #     cache_num=4*50, #cance to 4*50 for lung datast?
        #     cache_rate=cache_rate,
        #     num_workers=8,
        #     copy_cache=False,
        # )


        # # 2.5 s/it (16,16,16) ohne DA
        # train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=sample_batch_size, shuffle=True, repeats=1, drop_last=True)
        # #train_loader = ThreadBuffer(train_loader, 1)

        val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_num=6, cache_rate=cache_rate, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        #set_track_meta(True)
        return val_loader, val_ds


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def validation(epoch_iterator_val, patch_size, batch_size):

        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                # val_inputs[:, 1:2, :, :, :] = ((val_inputs[:, 1:2, :, :, :] * (250 - (-175)) + (-175) )/70 )
                # val_inputs[:, 1:2, :, :, :] = torch.clamp(val_inputs[:, 1:2, :, :, :], min=0.000 , max=1.000)
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model) 
                val_labels_list = decollate_batch(val_labels) 
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list] 
                
                val_outputs_list = decollate_batch(val_outputs) 
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list] 
                val_output_convert = [largest_connected(val_pred_tensor) for val_pred_tensor in val_output_convert] 
                    
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice_surface(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038bcbccc
            mean_dice_val = dice_metric.aggregate()
            mean_dice_surface = dice_surface.aggregate()
            dice_metric.reset()
            dice_surface.reset()
        return mean_dice_val, mean_dice_surface

    
    from torch.utils.tensorboard import SummaryWriter
    import shutil

    max_sample_size = 24  # dataset size for BTCV, 50 for lung ct
    num_classes = 14 # number of semantic classes, 14 for BTCV, is 2 for the lung_ct dataset
    torch.backends.cudnn.benchmark = True
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, batch=True, include_background=False,
                                            squared_pred=True)
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    dice_surface = SurfaceDiceMetric(class_thresholds = [1,1,1,1,1,1,1,1,1,1,1,1,1], include_background=False, reduction = "none")

    largest_connected = monai.transforms.KeepLargestConnectedComponent(num_components=1)

    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    from torch.optim.lr_scheduler import PolynomialLR, ExponentialLR, OneCycleLR, CyclicLR

        
    # Constants
    num_classes = 14  # 14 for BTCV, 2 for lung CT
    folds = 5  # Number of folds
    
    model_base_path = "/u/home/hodo/Documents/runs/LR_Baseline_"
    model_name = "UNet_final_Baseline_small_"

    # Store per-class dice values for all folds
    dice_vals_per_class_p_folds = []
    surface_dice_vals_per_class = []
    dice_vals_per_class = []
    for fold_idx in range(folds):
        print(f"Evaluating Fold {fold_idx}...")

        # Load the model for the current fold
        model_path = f"{model_base_path}{fold_idx}/{model_name}{fold_idx}.pth"
        # model = BasicUnetLocBAMs2(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=num_classes,
        #     features=(32, 32, 64, 128, 256, 32),
        #     dropout=0.1,

        #     hanet_params=None,
        # )

        model = BasicUNet(spatial_dims=3, 
                        in_channels=1, 
                        out_channels=14, 
                        features=(32, 32, 64, 128, 256, 32))

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # Define dataset parameters
        # patch_size = (128, 128, 128)
        # effective_batch_size = 2

        patch_size = (32, 32, 32)
        effective_batch_size = 128

        # Load validation dataset for the fold
        val_loader, val_ds = build_dataset(patch_size, effective_batch_size, crop_batch_size=1, fold_idx=fold_idx)

        # Perform validation
        epoch_iterator_val = tqdm(val_loader, desc=f"Validate Fold {fold_idx}", dynamic_ncols=True)
        dice_val, surface_dice = validation(epoch_iterator_val, patch_size, effective_batch_size)

        # Convert to numpy and store results
        dice_val_np = dice_val.cpu().numpy()#.mean(axis=0)  # Shape: (num_classes,)
        print(dice_val_np.shape, dice_val_np)
        # surface_dice_np = surface_dice.cpu().numpy().mean(axis=0)  # Shape: (num_classes,)
        # print(surface_dice_np.shape, surface_dice_np)
        # dice_vals_per_class_p_folds.append(dice_val_np) 
        dice_vals_per_class.append(dice_val_np)
        # print(dice_vals_per_class_p_folds) # Keep full per-fold data

        # surface_dice_vals_per_class.append(surface_dice_np)

    # print(dice_vals_per_class_p_folds)
    # Compute the average per-class Dice and Surface Dice across folds
    # average_dice_per_class = np.nanmean(dice_vals_per_class, axis=0).tolist()  # Shape: (num_classes,)
    # std_dice_per_class = np.nanstd(dice_vals_per_class, axis=0).tolist()  # Compute standard deviation
    # dice_vals_per_class = dice_vals_per_class.tolist()
    # average_surface_dice_per_class = np.nanmean(surface_dice_vals_per_class, axis=0).tolist()  # Shape: (num_classes,)
    # std_surface_dice_per_class = np.nanstd(surface_dice_vals_per_class, axis=0).tolist()  # Compute standard deviation

    # Save results to JSON files
    output_dir = "/u/home/hodo/Documents/"
    dice_output_file = os.path.join(output_dir, "Wilcoxon_dice_pc_Baseline_LC_SP")
    # surface_dice_output_file = os.path.join(output_dir, "sur_dice_pc_Baseline_SP.json")

    # Save Dice values (mean and std) to JSON
    # with open(dice_output_file, "w") as f:
    #     json.dump({
    #         "average_dice_per_class": average_dice_per_class,
    #         "std_dice_per_class": std_dice_per_class
    #     }, f, indent=4)
    with open(dice_output_file, "w") as f:
        json.dump({
            "per_fold_dice": [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dice_vals_per_class]
        }, f, indent=4)


    # # Save Surface Dice values (mean and std) to JSON
    # with open(surface_dice_output_file, "w") as f:
    #     json.dump({
    #         "average_surface_dice_per_class": average_surface_dice_per_class,
    #         "std_surface_dice_per_class": std_surface_dice_per_class
    #     }, f, indent=4)

    print(f"Results saved to {dice_output_file}") # and {surface_dice_output_file}")

        