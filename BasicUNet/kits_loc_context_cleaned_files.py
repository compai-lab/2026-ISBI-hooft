"""
# Copyright 2025 Donnate Bridget Hooft, Stefan Fischer
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)
# This script uses MONAI (Apache 2.0) for medical image processing and model training.

Description:
This script trains deep learning models for multi-organ segmentation on volumetric medical data (e.g., KiTS23).
It supports various architectural strategies including Baseline UNet, coordinate-augmented models, and LocBAM variants.
Training includes sliding window inference, mixed-precision training, polynomial LR scheduling, and coordinate-based augmentation.

Key features:
- 3D patch-based training with class-balanced sampling and dynamic patch sizing.
- Incorporates coordinate channels and body part regression (BPR) as additional supervision signals.
- Custom transforms such as `ConcatChannelsd` and `SplitChannelsd` enable modular, memory-efficient input handling.
- `OptimizeCoordinateChannels` and `ReconstructCoordinateChannels` reduce memory usage by compactly storing spatial coordinates.
- `BodyPartRegressorScoreTransform` rescales and extrapolates BPR scores to maintain anatomical continuity.
- **Corrupted or invalid image cases are automatically filtered out from training and validation to ensure robust data quality.**

Note:
These preprocessing enhancements help reduce memory overhead, accelerate I/O, and improve anatomical context modeling,
especially in hybrid image-feature representation settings.
"""


import os
import shutil
import tempfile

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
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, FlexibleUNet, UNETR, ViTAutoEnc, BasicUNet
#added code ______top
from monai.data import ITKReader, NibabelReader
from monai.transforms import LoadImaged
#added code ________ bottom
from monai.data import (
DataLoader,
    PersistentDataset,
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
from monai.transforms import MapTransform
from typing import Dict, Any
import sys
#sys.path.append('/vol/nasciam/datasets/amos_bpr/')
from LocBAM.LocBAM_1D import BasicUnetLocBAMs2

import torch.multiprocessing as mp
import torch
print(torch.cuda.is_available())
import argparse
#torch.multiprocessing.set_start_method("spawn")


class BodyPartRegressorScoreTransform(MapTransform):
    def __init__(self, keys: list):
        super().__init__(keys)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the transformation on the specified keys in the dictionary.
        """
        for key in self.keys:
            scores = data[key]["coord_z"]  # 1D tensor of body part regressor scores
            scores = self._linear_fitting_extrapolation(scores)
            data[key]["coord_z"] = scores
            print("cleaned scores: ", flush=True)
            print(scores, flush=True)
        return data

    def _linear_fitting_extrapolation(self, scores: torch.Tensor) -> torch.Tensor:
        """
        This function fits a linear model to the valid scores in [0, 100] range
        and uses it to extrapolate scores outside this range.
        """
        # Get indices of valid scores in the [0, 100] range
        valid_mask = (scores >= -0.5) & (scores <= 0.5)

        # Extract the valid values
        valid_scores = scores[valid_mask]

        if len(valid_scores) < 2:
            raise ValueError("At least two valid scores are required to perform linear fitting.")

        # Fit a linear function to the valid scores
        # Create x-values (indices) for the valid scores
        valid_indices = torch.nonzero(valid_mask).squeeze(dim=-1).float()

        # Fit a linear regression model: y = mx + b
        m, b = self._linear_fit(valid_indices, valid_scores)

        # Apply linear fitting to all values (both inside and outside the valid range)
        # For values within the valid range [0, 100], we retain the original values.
        scores_with_fitting = torch.where(
            valid_mask,
            scores,  # Keep valid scores as they are
            m * torch.arange(len(scores), dtype=torch.float32) + b  # Apply the linear fit to out-of-range scores
        )

        return scores_with_fitting

    def _linear_fit(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Perform linear fitting (y = mx + b) on the given x, y data points using least squares method.
        Returns the slope (m) and intercept (b).
        """
        # Calculate the slope (m) and intercept (b) using the least squares method
        A = torch.stack([x, torch.ones_like(x)], dim=-1)  # Create design matrix
        solution = torch.linalg.lstsq(A, y[:, None]).solution
        m = solution[0, 0]  # Slope
        b = solution[1, 0]  # Intercept
        return m, b

class ConcatChannelsd(MapTransform):
    """
    Concatenates multiple single-channel images into a multi-channel tensor.

    Example:
        Input: {"image_ct": (1, D, H, W), "image_other": (1, D, H, W)}
        Output: {"image": (2, D, H, W)}
    """

    def __init__(self, keys, output_key="image", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)

        # Ensure inputs are tensors
        tensors = [d[key] for key in self.keys]

        # Concatenate along channel dimension (dim=0)
        d[self.output_key] = torch.cat(tensors, dim=0)

        # Remove individual keys after concatenation
        for key in self.keys:
            del d[key]

        return d


class SplitChannelsd(MapTransform):
    """
    Splits a multi-channel image into separate dictionary keys.
    Assumes input has at least 2 channels.

    Example:
        Input: {"image": (2, D, H, W)}
        Output: {"image_ct": (1, D, H, W), "image_other": (1, D, H, W)}
    """

    def __init__(self, keys, output_keys=("image_ct", "image_other"), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_keys = output_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # Expected shape: (C, D, H, W)

            if isinstance(img, torch.Tensor):
                img = img.clone()  # Avoid modifying original tensor in-place

            d[self.output_keys[0]] = img[0:1]  # First channel
            d[self.output_keys[1]] = img[1:2]  # Second channel
            del d[key]  # Remove the original key to save memory

        return d


class OptimizeCoordinateChannels(MapTransform):
    """
    A MONAI transform that reduces memory usage by storing coordinate channels as 1D tensors.

    Assumes:
    - The first channel (index 0) is the CT image and remains unchanged.
    - The last three channels (index 1, 2, 3) are coordinate channels that change along only one axis.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # Shape: (4, W, H, D)  == ZXY
            # print(img.shape)
            # print(img[:,:3,:3,:3])
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Split channels
            ct_image = img[0:1]  # Shape: (1, D, H, W)
            # print(ct_image.shape)
            coord_x = img[2, :, 0, 0]  # Shape: (D,)
            coord_y = img[3, 0, :, 0]    # Shape: (H,)
            coord_z = img[1, 0, 0, :]   # Shape: (W,)

            print(coord_x[0])
            print(coord_x[-1])
            print()

            print(coord_y[0])
            print(coord_y[-1])
            print()

            print(coord_z[0])
            print(coord_z[-1])
            print()

            # Store as a dict for compact representation
            d[key] = {
                "ct_image": ct_image,        # (1, D, H, W)
                "coord_x": coord_x,          # (D,)
                "coord_y": coord_y,          # (H,)
                "coord_z": coord_z           # (W,)
            }
            # d["coord_z"]: coord_z  # (W,)

        return d

class ReconstructCoordinateChannels(MapTransform):
    """
    A MONAI transform that reconstructs the original (4, D, H, W) tensor from the optimized format.

    This is the inverse of OptimizeCoordinateChannels.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            compressed = d[key]

            # Retrieve compressed components
            ct_image = compressed["ct_image"]  # Shape: (1, D, H, W)
            coord_x = compressed["coord_x"]  # Shape: (D,)
            coord_y = compressed["coord_y"]  # Shape: (H,)
            coord_z = compressed["coord_z"]  # Shape: (W,)

            # Convert to tensors if needed
            if isinstance(coord_x, np.ndarray):
                coord_x = torch.from_numpy(coord_x)
            if isinstance(coord_y, np.ndarray):
                coord_y = torch.from_numpy(coord_y)
            if isinstance(coord_z, np.ndarray):
                coord_z = torch.from_numpy(coord_z)
            # print("here")
            # print(coord_x.shape, coord_y.shape, coord_z.shape)
            # print(coord_x[30:33])
            # Reshape and broadcast to full 3D shape
            coord_x = coord_x.view(-1, 1, 1).expand(ct_image.shape[1], ct_image.shape[2], ct_image.shape[3])  # (D, H, W)
            coord_y = coord_y.view(1, -1, 1).expand(ct_image.shape[1], ct_image.shape[2], ct_image.shape[3])  # (D, H, W)
            coord_z = coord_z.view(1, 1, -1).expand(ct_image.shape[1], ct_image.shape[2], ct_image.shape[3])  # (D, H, W)
            # print(coord_x.shape, coord_y.shape, coord_z.shape)
            # Stack coordinate channels back into the full tensor
            full_tensor = torch.cat(
                [ct_image, coord_z.unsqueeze(0), coord_x.unsqueeze(0), coord_y.unsqueeze(0)],
                dim=0
            )
            # print(full_tensor.shape)
            # Store the restored tensor
            d[key] = full_tensor  # Shape: (4, D, H, W)

        return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fold", type=int, choices=[0,1,2,3,4])
    parser.add_argument("-strategy", choices=["Baseline", "XYZ", "LocBAM"])
    parser.add_argument("-ps", choices=["small", "large"])

    args = parser.parse_args()

    dtype = [np.float16, np.int8]
    

    mp.set_start_method("spawn")
    print_config()
    # the train dataset and the cross-validation split configs are here
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    num_classes = 4 # amos 16 #14 for btcv dataset, 4kits
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_train_transforms(patch_size, crop_batch_size, sampling):
        # class oversampling with 50% background (class_idx=0) and 50% foreground (class_idx!=0)
        class_ratio = np.ones(num_classes)
        class_ratio[0] = num_classes - 1
        max_class_indizes = 10000  # for fast dataloading

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                SplitChannelsd(keys=["image"], output_keys=("image_ct", "image_bpr")),
                ScaleIntensityRanged(
                    keys=["image_ct"],
                    a_min=-58,
                    a_max=302,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["image_bpr"],
                    a_min=0,
                    a_max=100,
                    b_min=-0.5,
                    b_max=0.5,
                    clip=False,
                ),
                ConcatChannelsd(keys=["image_ct", "image_bpr"], output_key="image"),

                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),

                AddCoordinateChannelsd(keys=["image"], spatial_dims=[0, 1]),

                ClassesToIndicesd(  # for fast dataloading
                    keys=["image", "label"],
                    image_key="image",
                    num_classes=num_classes,
                    indices_postfix="_cls",
                    max_samples_per_class=max_class_indizes
                ),
                OptimizeCoordinateChannels(
                    keys=["image"]
                ),
                BodyPartRegressorScoreTransform(keys=["image"]),
                RandShiftIntensityd(
                    keys=["image"],
                    prob=0.0,
                    offsets=0.0
                ),
                ReconstructCoordinateChannels(
                    keys=["image"]
                ),
                # here we create patches from full volumes
                EnsureTyped(keys=["image", "label"], track_meta=False, dtype=dtype),
                EnsureTyped(keys=["image", "label"], device=device, track_meta=False, dtype=dtype),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    num_classes=num_classes,
                    ratios=class_ratio,
                    spatial_size=patch_size,
                    num_samples=crop_batch_size,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True,
                    indices_key="label_cls",
                    max_samples_per_class=max_class_indizes
                ),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=patch_size
                ),

                ToTensord(
                    keys=["image", "label"],
                    track_meta=False
                ),
                EnsureTyped(keys=["image", "label"], device=device, track_meta=False, dtype=dtype),

            ]
        )

        return train_transforms


    def get_val_transforms(sampling):

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                SplitChannelsd(keys=["image"], output_keys=("image_ct", "image_bpr")),
                ScaleIntensityRanged(
                    keys=["image_ct"],
                    a_min=-58,
                    a_max=302,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["image_bpr"],
                    a_min=0,
                    a_max=100,
                    b_min=-0.5,
                    b_max=0.5,
                    clip=False,
                ),
                ConcatChannelsd(keys=["image_ct", "image_bpr"], output_key="image"),

                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),
                # add chanmel for correct scores here
                AddCoordinateChannelsd(keys=["image"], spatial_dims=[0, 1]),
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1,2]),
                OptimizeCoordinateChannels(
                    keys=["image"]
                ),
                BodyPartRegressorScoreTransform(keys=["image"]),
                RandShiftIntensityd(
                    keys=["image"],
                    prob=0.0,
                    offsets=0.0
                ),
                ReconstructCoordinateChannels(
                    keys=["image"]
                ),
                EnsureTyped(keys=["image", "label"], track_meta=False, dtype=dtype),
                EnsureTyped(keys=["image", "label"], device=device, track_meta=False, dtype=dtype),

            ]
        )

        return val_transforms


    def build_dataset(current_patch_size, sample_batch_size, crop_batch_size, fold_idx, sampling):

        data_dir = "/home/iml/stefan.fischer/transfer-code/kits-jsons/"  ## <-- CHANGE THIS TO YOUR LOCAL DATA LOCATION

        # copy to aims cluster

        split_json = "dataset_" + str(fold_idx) + "_hmcluster_bpr_new.json"

        datasets = data_dir + split_json
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")

        wrong_file_ids = ["00516", "00223", "00088", "00560"]
        # filter val list with wrong file ids
        cleaned_val_files = []
        for case in val_files:
            clean_file = True
            for error_id in wrong_file_ids:
                if error_id in case["image"]:
                    print(case["image"])
                    print("skipping val case : " + str(error_id))
                    clean_file = False
            if clean_file:
                cleaned_val_files.append(case)

        # filter train list with wrong file ids
        cleaned_train_files = []
        for case in datalist:
            clean_file = True
            for error_id in wrong_file_ids:
                if error_id in case["image"]:
                    print(case["image"])
                    print("skipping train case : " + str(error_id))
                    clean_file = False
            if clean_file:
                cleaned_train_files.append(case)


        cache_rate = 1.0
        if True:
            train_ds = CacheDataset(
                data=cleaned_train_files,
                transform=get_train_transforms(current_patch_size, crop_batch_size, sampling),
                cache_rate=cache_rate,
                num_workers=8,
                copy_cache=False,
            )
        else:
            train_ds = PersistentDataset(
                data=datalist,
                transform=get_train_transforms(current_patch_size, crop_batch_size, sampling),
                #cache_dir="/tmp/train"
                # cache_dir=  "/vol/ciamspace/datasets/amos_bpr/tmp/"
                cache_dir=  "/lustre/boost_ai/users/test/stefan.fischer/nnUNet_raw/Dataset012_KiTS23/cache/"
            )
        import os
        os.system("nvidia-smi")
        # train_ds = SmartCacheDataset(
        #     data=datalist,
        #     transform=get_train_transforms(current_patch_size, crop_batch_size, sampling),
        #     replace_rate=0.1,  # Adjust as needed (10% of data replaced per epoch)
        #     cache_num=min(len(datalist), 200),  # Cache part of the dataset
        #     cache_rate=0.1,  # Modify based on RAM availability
        #     num_init_workers=4,
        #     num_replace_workers=2,
        #     shuffle=True,
        #     copy_cache=False,  # Avoid modifying cached data
        # )
        
        # 2.5 s/it (16,16,16) ohne DA
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=sample_batch_size, shuffle=True, repeats=1, drop_last=True)
        train_loader = ThreadBuffer(train_loader, 1)
        cache_rate = 0.0
        if True:
            val_ds = CacheDataset(data=cleaned_val_files, transform=get_val_transforms(sampling), cache_rate=cache_rate, num_workers=4)
        else:
            val_ds =PersistentDataset(
                data=val_files,
                transform=get_val_transforms(sampling),
                #cache_dir="/tmp/train"
                # cache_dir= "vol/ciamspace/datasets/amos_bpr/tmp/"

                cache_dir=  "/lustre/boost_ai/users/test/stefan.fischer/nnUNet_raw/Dataset012_KiTS23/cache/"
            )
        # val_ds = CacheDataset(data=val_files, transform=get_val_transforms(sampling), cache_num=10, cache_rate=cache_rate, num_workers=4)
        # val_ds =SmartCacheDataset(
        #     data=val_files,
        #     transform=get_val_transforms(sampling),
        #     replace_rate=0.1,  # Adjust as needed (10% of data replaced per epoch)
        #     cache_num=10,  # Cache part of the dataset
        #     cache_rate=0.25,  # Modify based on RAM availability
        #     num_init_workers=4,
        #     num_replace_workers=2,
        #     shuffle=True,
        #     copy_cache=False,  # Avoid modifying cached data
        # )
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        #set_track_meta(True)
        return train_loader, val_loader, train_ds, val_ds


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def validation(epoch_iterator_val, patch_size, batch_size, sampling, max_samples=None):
        model.eval()
        dice_metric.reset()
        
        sample_count = 0  # Track number of samples processed
        
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

                # Apply preprocessing depending on sampling method
                if sampling == "Baseline":
                    val_inputs = val_inputs[:, 0:1, :, :, :]

                # Run inference
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model, device=torch.device("cpu"))

                # Compute Dice metric
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor.cpu()) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor.cpu()) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

                sample_count += len(val_inputs)  # Track how many samples were processed
                
                # Stop early if max_samples is set
                if max_samples is not None and sample_count >= max_samples:
                    break

            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()

        return mean_dice_val



    from torch.utils.tensorboard import SummaryWriter
    import shutil

    for LocContext in [args.strategy]: #"Baseline", "XYZ", #, "XYZ", "LocBAM"#adjust "Baseline",  for conidtions here # , "BPRXY",
        for p_size in [args.ps]:# ,"large" , "small"
            for fold_idx in [args.fold]:
            #for fold_idx in range(5):
                if LocContext == "Baseline":
                    sampling = "Baseline"
                    nr_in_channels = 1
                elif LocContext == "XYZ":
                    sampling = "XYZ"
                    nr_in_channels = 4
                elif LocContext == "LocBAM":
                    sampling = "XYZ"
                    nr_in_channels = 1
                else:
                    raise ValueError(f"Unknown LocContext: {LocContext}")

                print(" using strategy" , LocContext)

                max_sample_size = 2  
                torch.backends.cudnn.benchmark = True
                loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, batch=True, include_background=False,
                                                        squared_pred=True)
                post_label = AsDiscrete(to_onehot=num_classes)
                post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
                dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

                global_step = 0
                dice_val_best = 0.0
                global_step_best = 0
                epoch_loss_values = []
                metric_values = []
                from torch.optim.lr_scheduler import PolynomialLR, ExponentialLR, OneCycleLR, CyclicLR

                            
                if p_size == "small":
                        patch_size = (32, 32, 32)
                        effective_batch_size = 128
                    # patch_size = (64, 64, 64)
                    # effective_batch_size = 16
                elif p_size == "large":
                    patch_size = (128, 128, 128)
                    effective_batch_size = 2

                else:
                    print("No patch size selected") 
                    # Calculate the number of parameters

                # max_iterations = 250000 #number of training iterations
                max_iterations = 250000
                eval_num = max_iterations // 20 #when to trigger evaluation during training

                new_lr = 1e-2
                if LocContext == "LocBAM":
                    model = BasicUnetLocBAMs2(
                            spatial_dims=3,
                            in_channels=1,
                            out_channels=num_classes,
                            features=(32, 32, 64, 128, 256, 32),
                            dropout=0.1,
                            hanet_params=None,
                        )
                else:
                    model = BasicUNet(spatial_dims=3, 
                        in_channels=nr_in_channels, 
                        out_channels=num_classes, 
                        features=(32, 32, 64, 128, 256, 32))

                optimizer = torch.optim.SGD(model.parameters(), lr=new_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
                # optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=3e-5)
                lr_scheduler = PolynomialLR(optimizer, total_iters=1000, power=0.9)

                # for mixed precision training (faster training)
                scaler = torch.cuda.amp.GradScaler()

                    # pytorch_total_params = sum(p.numel() for p in model.parameters())
                # print("Total num of parameters " + str(pytorch_total_params))

                model = model.to(device)

                ### train
                writer = SummaryWriter()
                logdir = writer.get_logdir() #wha tis logdir in cluster
                # logdir = "/vol/nasciam/users/hodo/Documents/Cluster/runs"
                print(logdir)
                shutil.copy2(__file__, logdir)

                # Inter sample cropping (sample>=2, crops_per_sample>=1)
                if effective_batch_size > max_sample_size:
                    sample_batch_size = max_sample_size
                    crop_batch_size = int(np.ceil(effective_batch_size / sample_batch_size))
                else:
                    crop_batch_size = 1
                    sample_batch_size = effective_batch_size

                train_loader, val_loader, train_ds, val_ds = build_dataset(patch_size, sample_batch_size,
                                                                        crop_batch_size, fold_idx, sampling)

                # train_ds.start()  # Start caching
                while global_step <= max_iterations:
                    writer.add_scalar("batch_size/crop_batch_size", crop_batch_size, global_step)
                    writer.add_scalar("batch_size/effective_batch_size", effective_batch_size, global_step)
                    writer.add_scalar("batch_size/sample_batch_size", sample_batch_size, global_step)


                    model.train()
                    epoch_loss = 0
                    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
                    for step, batch in enumerate(epoch_iterator):
                        global_step += 1
                        if global_step >= max_iterations:
                            break

                        step += 1 # to avoid step = 0!

                        x, y = (batch["image"].cuda(), batch["label"].cuda())

                        # accces inidvidual channels between value rage (0-1)

                        if effective_batch_size < crop_batch_size*sample_batch_size:
                            # dirty: use only num of effective batch size samples
                            x = x[:effective_batch_size]
                            y = y[:effective_batch_size]

                        if LocContext == "Baseline":
                            x = x[:, 0:1, :, :, :]

                        print(x.shape)
                        
                        # patch_values = x[1,:,:3,:3,:3]
                        # print("patchv values =", patch_values)
                        
                        with torch.cuda.amp.autocast():
                            logit_map = model(x)
                            # print("here1")
                            loss = loss_function(logit_map, y)
                            # print("here2")

                        ##nnunet version (w gradient clipping)
                        scaler.scale(loss).backward()
                        # print("here3")
                        scaler.unscale_(optimizer)
                        # print("here4")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                        # print("here5")
                        # normal = 12, tried 8 / 6 for alrger patch sizes
                        scaler.step(optimizer)
                        # print("here6")
                        scaler.update()
                        # print("here7")
                        epoch_loss += loss.item()
                        # print("here8")
                        optimizer.zero_grad(set_to_none=True)
                        # print("here9")

                        factor = max_iterations/ 1000
                        if global_step % factor == 0: # should result in 1000 epochs
                            if lr_scheduler:
                                print("LR scheduler lr : " + str(lr_scheduler.get_last_lr()))
                                lr_scheduler.step()


                        epoch_iterator.set_description(  # noqa: B038
                            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
                        )
                        writer.add_scalar("Loss/train", loss, global_step)
                        writer.add_scalar("Patchsize/train", patch_size[0], global_step)

                        if (global_step % eval_num == 0 and global_step != 0):
                            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

                            # Validate only on 10 samples for intermediate steps
                            dice_val = validation(epoch_iterator_val, patch_size, effective_batch_size, sampling, max_samples=10)

                            epoch_loss /= step
                            epoch_loss_values.append(epoch_loss)
                            metric_values.append(dice_val)

                            writer.add_scalar("Loss/val", epoch_loss, global_step)
                            writer.add_scalar("Dice/val", dice_val, global_step)

                        # if global_step % 1000 == 0:  # Update cache less often
                        #     train_ds.update_cache() 

                        # Final validation step (on the full dataset)
                        if global_step == max_iterations:
                            epoch_iterator_val = tqdm(val_loader, desc="Validate (Final Full Dataset)", dynamic_ncols=True)

                            # Validate on **all** validation samples at the final step
                            dice_val = validation(epoch_iterator_val, patch_size, effective_batch_size, sampling, max_samples=None)

                            epoch_loss /= step
                            epoch_loss_values.append(epoch_loss)
                            metric_values.append(dice_val)

                            writer.add_scalar("Loss/val", epoch_loss, global_step)
                            writer.add_scalar("Dice/val", dice_val, global_step)



                            if dice_val > dice_val_best:
                                dice_val_best = dice_val
                                global_step_best = global_step
                                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                                print(
                                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                                )
                            else:
                                print(
                                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                                        dice_val_best, dice_val
                                    )
                                )
                        if global_step >= max_iterations:
                            break

                    writer.flush()
                    if global_step >= max_iterations:
                        break


                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val, patch_size, effective_batch_size, sampling, max_samples=None)

                metric_values.append(dice_val)
                writer.add_scalar("Dice/val", dice_val, global_step)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
                print(f"train completed, last_metric: {dice_val:.4f} " f"at iteration: {max_iterations}")

                writer.flush()
                writer.close()
                performance_string = str(np.round(dice_val, decimals=4))

                # torch.save(model.state_dict(), os.path.join(logdir, "UNet_final_model_" + str(LocContext)+"_"+str(fold_idx) +".pth"))
                torch.save(model.state_dict(), os.path.join(logdir, "cleaned_KITS23_UNet_final_" + str(LocContext)+"_"+str(p_size)+"_"+str(fold_idx)  + "_" + str(performance_string) +".pth"))