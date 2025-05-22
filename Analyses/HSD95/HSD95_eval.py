"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (see LICENSE file or http://www.apache.org/licenses/LICENSE-2.0)
# This file includes components adapted from MONAI (Apache 2.0).
"""

# For the BTCV dataset, this code iterates over 5 cross-validation folds created for each model type.
# It loads the corresponding model for each fold and evaluates it on that fold.
# The final results for HSD (Hausdorff Distance) are reported as the average over the 5 folds,
# along with the standard deviation.

import os
import shutil
import tempfile
import json

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
from monai.metrics import DiceMetric, SurfaceDiceMetric, HausdorffDistanceMetric
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
from BUNet.LocBAM.LocBAM_1D import BasicUnetHANets2

from UNETR_alternative_PE import UNETRCoordEmbed

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
       

        val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_num=6, cache_rate=cache_rate, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        #set_track_meta(True)
        return val_loader, val_ds


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def validation(epoch_iterator_val, patch_size, batch_size):
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model)

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(v) for v in val_labels_list]

                val_outputs_list = decollate_batch(val_outputs)
                val_outputs_convert = [post_pred(v) for v in val_outputs_list]
                val_outputs_convert = [largest_connected(v) for v in val_outputs_convert]

                hd95_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                torch.cuda.empty_cache()
                epoch_iterator_val.set_description(f"Evaluating HD95")
               

            hd95_value = hd95_metric.aggregate().item()
            hd95_metric.reset()
        return hd95_value

    
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
    
    model_base_path = "/u/home/hodo/Documents/Cluster/runs/BTCV_UNETR_BP_"
    model_name = "Final_UNETR_large_"
    # Cluster/runs/BTCV_UNETR_0/Final_UNETR_small_0.pth
    # 0/_0.pth

    # Store per-class dice values for all folds
    dice_vals_per_class_p_folds = []
    surface_dice_vals_per_class = []
    dice_vals_per_class = []
   
    hd95_scores = []
    for fold_idx in range(folds):
        print(f"Evaluating Fold {fold_idx}...")

        model_path = f"{model_base_path}{fold_idx}/{model_name}{fold_idx}.pth"
        
        # model = BasicUnetLocBAMs2(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=num_classes,
        #     features=(32, 32, 64, 128, 256, 32),
        #     dropout=0.1,

        #     hanet_params=None,
        # )
        img_size =128

        model =  UNETR(
                        in_channels=1,          # Number of input channels (e.g., grayscale image with 1 channel)
                        out_channels=14,        # Number of output channels (e.g., for segmentation tasks, this could be 14 classes)
                        img_size=img_size            # Input image size (32x32x32 for 3D or 32x32 for 2D)
                    )
        
        # model = UNETRCoordEmbed(in_channels=1, out_channels=num_classes, img_size=img_size) #.to("cuda" if torch.cuda.is_available() else "cpu")


        # model = BasicUNet(spatial_dims=3, 
        #                 in_channels=1, 
        #                 out_channels=14, 
        #                 features=(32, 32, 64, 128, 256, 32))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
        # model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        # patch_size = (32, 32, 32)
        # effective_batch_size = 128
        patch_size = (128, 128, 128)
        effective_batch_size = 2
        val_loader, val_ds = build_dataset(patch_size, effective_batch_size, crop_batch_size=1, fold_idx=fold_idx)
        epoch_iterator_val = tqdm(val_loader, desc=f"Validate Fold {fold_idx}", dynamic_ncols=True)

        hd95 = validation(epoch_iterator_val, patch_size, effective_batch_size)
        print(f"Fold {fold_idx} HD95: {hd95:.4f}")
        hd95_scores.append(hd95)

    hd95_mean = np.mean(hd95_scores)
    hd95_std = np.std(hd95_scores)
    print(f"\nAverage HD95 across 5 folds: {hd95_mean:.4f}")
    print(f"Standard deviation of HD95: {hd95_std:.4f}")

   