
#testing differemt locationcontexts, Baseline, Z , XYZ, only basicunet
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

import sys
sys.path.append('/vol/nasciam/datasets/amos_bpr/') # <-- CHANGE THIS TO YOUR LOCAL DATA LOCATION
from Alt_PE.UNETR_alternative_PE import UNETRCoordEmbed

import torch.multiprocessing as mp
import torch
import torch
orig_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)

torch.load = patched_load

print(torch.cuda.is_available())
import argparse
#torch.multiprocessing.set_start_method("spawn")


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fold", type=int, choices=[0,1,2,3,4])
    args = parser.parse_args()
    
    args = parser.parse_args()

    dtype = [np.float16, np.int8]
    

    mp.set_start_method("spawn")
    print_config()
    # the train dataset and the cross-validation split configs are here
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    num_classes = 16 #14 for btcv dataset, 2 for lung_dataset
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
                monai.transforms.DeleteItemsd(
                    keys=["image_bpr"],
                ),
                ScaleIntensityRanged(
                    keys=["image_ct"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image_ct", "label"], source_key="image_ct"),
                Orientationd(keys=["image_ct", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image_ct", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),

                ClassesToIndicesd(  # for fast dataloading
                    keys=["image_ct", "label"],
                    image_key="image_ct",
                    num_classes=num_classes,
                    indices_postfix="_cls",
                    max_samples_per_class=max_class_indizes
                ),
                EnsureTyped(keys=["image_ct", "label"], device=torch.device("cpu"), track_meta=False, dtype=dtype),
                # here we create patches from full volumes
                RandCropByLabelClassesd(
                    keys=["image_ct", "label"],
                    label_key="label",
                    num_classes=num_classes,
                    ratios=class_ratio,
                    spatial_size=patch_size,
                    num_samples=crop_batch_size,
                    image_key="image_ct",
                    image_threshold=0,
                    allow_smaller=True,
                    indices_key="label_cls",
                    max_samples_per_class=max_class_indizes
                ),
                SpatialPadd(
                    keys=["image_ct", "label"],
                    spatial_size=patch_size
                ),

                ToTensord(
                    keys=["image_ct", "label"],
                    track_meta=False
                ),

            ]
        )

        return train_transforms


    def get_val_transforms(sampling):

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                SplitChannelsd(keys=["image"], output_keys=("image_ct", "image_bpr")),
                monai.transforms.DeleteItemsd(
                    keys=["image_bpr"],
                ),
                ScaleIntensityRanged(
                    keys=["image_ct"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),

                CropForegroundd(keys=["image_ct", "label"], source_key="image_ct", allow_smaller=False),
                Orientationd(keys=["image_ct", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image_ct", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                ),
                # add chanmel for correct scores here
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1,2]),
                EnsureTyped(keys=["image_ct", "label"], device=torch.device("cpu"), track_meta=False, dtype=dtype),

            ]
        )

        return val_transforms

    def build_dataset(current_patch_size, sample_batch_size, crop_batch_size, fold_idx, sampling):
       
        
        data_dir = "/vol/nasciam/datasets/amos_bpr/" # <-- CHANGE THIS TO YOUR LOCAL DATA LOCATION

            # copy to aims cluster 
        
        split_json = "dataset_cluster_" + str(fold_idx) + ".json"

        datasets = data_dir + split_json
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        cache_rate = 1.0


        cache_dir = "/vol/nasciam/datasets/amos_bpr/tmp/cache_baseline" # <-- CHANGE THIS TO YOUR LOCAL DATA LOCATION
        if False:
            train_ds = CacheDataset(
                data=datalist,
                transform=get_train_transforms(current_patch_size, crop_batch_size, sampling),
                cache_rate=cache_rate,
                num_workers=8,
                copy_cache=False,
            )
        else:
            train_ds = PersistentDataset(
                data=datalist,
                transform=get_train_transforms(current_patch_size, crop_batch_size, sampling),
                
                cache_dir= cache_dir #, 
                # allow_cache_existing=True
            )

        
        # 2.5 s/it (16,16,16) ohne DA
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=sample_batch_size, shuffle=True, repeats=1, drop_last=True, pin_memory=True)
        train_loader = ThreadBuffer(train_loader, 1)
        if False:
            val_ds = CacheDataset(data=val_files, transform=get_val_transforms(sampling), cache_rate=cache_rate, num_workers=4)
        else:
            val_ds =PersistentDataset(
                data=val_files,
                transform=get_val_transforms(sampling),
                #cache_dir="/tmp/train"
                # cache_dir= "vol/ciamspace/datasets/amos_bpr/tmp/"

                cache_dir=  cache_dir #, #"/vol/nasciam/datasets/amos_bpr/cache/"
                # allow_cache_existing=True
            )
       
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, pin_memory=True)
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
                val_inputs, val_labels = (batch["image_ct"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True))

                # Run inference
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model,
                                                           device=torch.device("cpu"))

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

    for LocContext in ["UNETR"]:#,  "Baseline", "XYZ", "HAN2",#adjust for conidtions here # , "BPRXY",
        for p_size in [ "small", "large"]:# ,, "small" ,"large" 
            for fold_idx in [args.fold]:
            #for fold_idx in range(5):
                if LocContext == "UNETR":
                    sampling = "Baseline"
                    nr_in_channels = 1
                elif LocContext == "XYZ":
                    sampling = "XYZ"
                    nr_in_channels = 4
                elif LocContext == "HAN2":
                    sampling = "XYZ"
                    nr_in_channels = 1
                else:
                    raise ValueError(f"Unknown LocContext: {LocContext}")

                print(" using strategy" , LocContext)

                max_sample_size = 2  # dataset size for BTCV, 50 for lung ct
                num_classes = 16 # number of semantic classes, 14 for BTCV, is 2 for the lung_ct dataset
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
                    img_size=32
                    # patch_size = (64, 64, 64)
                    # effective_batch_size = 16
                elif p_size == "large":
                    patch_size = (128, 128, 128)
                    effective_batch_size = 2
                    img_size=128

                else:
                    print("No patch size selected") 
                    # Calculate the number of parameters

                # max_iterations = 250000 #number of training iterations
                max_iterations = 250000
                eval_num = max_iterations // 20 #when to trigger evaluation during training

                # new_lr = 1e-2
                new_lr = 1e-4

                if LocContext == "UNETR":
                    model = model = UNETR(
                        in_channels=1,          # Number of input channels (e.g., grayscale image with 1 channel)
                        out_channels=num_classes,        # Number of output channels (e.g., for segmentation tasks, this could be 14 classes)
                        img_size=img_size            # Input image size (32x32x32 for 3D or 32x32 for 2D)
                    )

                elif LocContext == "UNETR_ALT_PE":
                    model = UNETRCoordEmbed(in_channels=1, out_channels=num_classes, img_size=img_size) #.to("cuda" if torch.cuda.is_available() else "cpu")

                # optimizer = torch.optim.SGD(model.parameters(), lr=new_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-5)

                # optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=3e-5)
                # lr_scheduler = PolynomialLR(optimizer, total_iters=1000, power=0.9)
                lr_scheduler = None
                # for mixed precision training (faster training)
                scaler = torch.cuda.amp.GradScaler()

                    # pytorch_total_params = sum(p.numel() for p in model.parameters())
                # print("Total num of parameters " + str(pytorch_total_params))

                model = model.to(device)
                model = torch.compile(model)

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

                        x, y = (batch["image_ct"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True))

                    

                        if effective_batch_size < crop_batch_size*sample_batch_size:
                            # dirty: use only num of effective batch size samples
                            x = x[:effective_batch_size]
                            y = y[:effective_batch_size]
                        
                        with torch.cuda.amp.autocast():
                            logit_map = model(x)
                            loss = loss_function(logit_map, y)

                        ##nnunet version (w gradient clipping)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    
                        # normal = 12, tried 8 / 6 for alrger patch sizes
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_loss += loss.item()
                        optimizer.zero_grad(set_to_none=True)

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
                torch.save(model.state_dict(), os.path.join(logdir, "AMOS22_final_" + str(LocContext)+"_"+str(p_size)+"_"+str(fold_idx)+ "_" + str(performance_string) +".pth"))