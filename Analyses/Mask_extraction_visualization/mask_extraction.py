"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

Description:
This script is designed for extracting and saving segmentation masks from trained 3D medical image segmentation models 
(e.g., UNETR, BasicUNet) using the MONAI framework. It supports inference on volumetric NIfTI images using a 
sliding window approach, and saves both predicted segmentation masks and corresponding input CT scans for downstream 
visualization or analysis.

Key features:
- Supports MONAI-compatible models trained on datasets such as KiTS, BTCV, and AMOS.
- Uses `sliding_window_inference` for memory-efficient evaluation on large 3D volumes.
- Post-processes predictions with connected component filtering.
- Saves both predictions and inputs as `.nii.gz` files using SimpleITK.
- Ideal for qualitative assessment or visual debugging of trained models.

Note:
Make sure the provided model path and dataset configuration (e.g., image sizes, folds) match the format used during training.
"""

import os
import shutil
import tempfile
import json

import matplotlib.pyplot as plt
import monai.losses
import numpy as np
from tqdm import tqdm
from SimpleITK import write_nifti 

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
import nibabel
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
from UNETR_alternative_PE import UNETRCoordEmbed
import torch.multiprocessing as mp
import torch

from monai.data import set_track_meta
set_track_meta(True)
#torch.multiprocessing.set_start_method("spawn")
if __name__ == "__main__":
    mp.set_start_method("spawn")
    print_config()
    # the train dataset and the cross-validation split configs are here
    # os.environ["MONAI_DATA_DIRECTORY"] = "/home/compai/code/data/nnunet_raw/nnunet_dataset/Dataset101_BTCV_abdomen/data/btcv/"
    
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)


    num_classes = 16 #14 for btcv dataset, 2 for lung_dataset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #

    def get_val_transforms():
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
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
                
                EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
            ]
        )
        return val_transforms


    def build_dataset(patch_size, batch_size, crop_batch_size, fold_idx, max_samples):
        data_dir = "/vol/ciamspace/datasets/KiTs23/dataset/"
        split_json = "random_samples.json"
        datasets = os.path.join(data_dir, split_json)
        val_files = load_decathlon_datalist(datasets, True, "validation")
        val_files = val_files[1:max_samples+1]
 
        val_ds = CacheDataset(
            data=val_files,
            transform=get_val_transforms(),
            cache_rate=0.333,
            num_workers=4
        )
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        return val_loader, val_ds

    # def build_dataset(patch_size, batch_size, crop_batch_size, fold_idx, max_samples=6):
    #     data_dir = "/vol/ciamspace/datasets/amos_bpr/"
    #     split_json = f"dataset_{fold_idx}.json"
    #     datasets = os.path.join(data_dir, split_json)
    #     val_files = load_decathlon_datalist(datasets, True, "validation")

    #     val_files = val_files[:max_samples]  # Limit number of samples loaded

    #     val_ds = CacheDataset(
    #         data=val_files,
    #         transform=get_val_transforms(),
    #         cache_rate=0.2,
    #         num_workers=2  # Reduce workers to lower RAM usage
    #     )
    #     val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    #     return val_loader, val_ds

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def validation(epoch_iterator_val, patch_size, batch_size):

        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_inputs = val_inputs[:, 0:1, :, :, :]

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

    # from monai.data import write_nifti

    def validation_single_sample(val_loader, patch_size, batch_size, output_path):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i > 0:
                    break  # Only process the first sample

                val_inputs = batch["image"].to(device)
                val_labels = batch["label"].to(device)

                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model)

                # Post-processing
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                val_output_convert = [largest_connected(val_pred_tensor) for val_pred_tensor in val_output_convert]

                # Save prediction as .nii (assumes batch_size = 1)
                meta = batch["image"].meta_dict
                write_nifti(
                    data=val_output_convert[0].cpu(),
                    file_name=output_path,
                    affine=meta["original_affine"],
                    target_affine=meta.get("affine", None),
                    output_spatial_shape=meta.get("spatial_shape", None),
                    mode="nearest"
                )

                print(f"Saved prediction to {output_path}")
                break  # Only one sample


    from torch.utils.tensorboard import SummaryWriter
    import shutil

    max_sample_size = 24  # dataset size for BTCV, 50 for lung ct
    num_classes = 16 # number of semantic classes, 14 for BTCV, is 2 for the lung_ct dataset
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
    num_classes = 4  # 4 for KiTS, 14 for BTCV, 16 for AMOS
    folds = 5  # Number of folds


    from monai.transforms import SaveImage

    # Set up SaveImage transform
    save_output = SaveImage(
        output_dir="Documents",  # Change this to your desired path
        output_postfix="pred",        # Appended to base filename
        output_ext=".nii.gz",         # Output format
        separate_folder=False,        # All saved in the same output dir
        resample=False,               # Keep original spacing/orientation
        print_log=True,
        output_dtype=np.uint8,        # Keep labels as integers (e.g., 1, 2, 3...)
        squeeze_end_dims=True
    )

    # Just one fold and one sample for demonstration
    fold_idx = 1
    print(f"Evaluating ONE sample from Fold {fold_idx}...")
    import torch
    import numpy as np
    import SimpleITK as sitk
    from monai.inferers import sliding_window_inference
    from monai.data import decollate_batch
    from monai.networks.nets import BasicUNet

    # Load model
    model_path = "/u/home/hodo/Documents/locunet/runs/KITS/KITS23_UNet_final_Baseline_large_0/KITS23_UNet_final_Baseline_large_0.pth"
    img_size=32
    # model =  UNETR(
    #                     in_channels=1,          # Number of input channels (e.g., grayscale image with 1 channel)
    #                     out_channels=14,        # Number of output channels (e.g., for segmentation tasks, this could be 14 classes)
    #                     img_size=img_size            # Input image size (32x32x32 for 3D or 32x32 for 2D)
    #                 )
    # model = BasicUNet(spatial_dims=3, 
    #                     in_channels=1, 
    #                     out_channels=num_classes, 
    #                     features=(32, 32, 64, 128, 256, 32)).to(device)
    
    img_size = 32
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=img_size
    ).to(device)
    # model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=14, features=(32, 32, 64, 128, 256, 32))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)

    model = model.to(device)
    model.eval()

    # Set patch size and batch size
    patch_size = (32, 32, 32)
    effective_batch_size = 128

    # Load dataset
    val_loader, val_ds = build_dataset(patch_size, effective_batch_size, crop_batch_size=1, fold_idx=1, max_samples=1)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx > 0:
                break  # Only process one sample

            val_inputs = batch["image"].to(device)
            val_inputs = val_inputs[:, 0:1, :, :, :]
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, patch_size, 1, model)

            val_outputs_list = decollate_batch(val_outputs)
            val_preds = [torch.argmax(pred, dim=0) for pred in val_outputs_list]

            # Save prediction and input image
            pred_np = val_preds[0].cpu().numpy().astype(np.uint8)
            input_np = val_inputs[0][0].cpu().numpy().astype(np.float32)

            sitk.WriteImage(sitk.GetImageFromArray(pred_np), "prediction_kits_untr.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(input_np), "ct_kits_untr.nii.gz")

            print("✅ Saved prediction.nii.gz and ct.nii.gz")
