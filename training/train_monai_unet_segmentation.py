#!/usr/bin/python

from __future__ import print_function
import sys
import os
import shutil
import tempfile
import torch
from tqdm import tqdm

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
#from tqdm import tqdm
import nibabel as nib 

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandHistogramShiftd,
    RandAffined,
    ToTensord,  
)


from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet, AttentionUnet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

import warnings
warnings.filterwarnings("ignore")


torch.cuda.empty_cache()

#############################################################################################################
#############################################################################################################



files_path = sys.argv[1]
check_path = sys.argv[2]
json_file = sys.argv[3]
results_path = sys.argv[4]

res = int(sys.argv[5])
cl_num = int(sys.argv[6])

status_train_proc = int(sys.argv[7])
status_load_check = int(sys.argv[8])
max_iterations = int(sys.argv[9])
roi_type = sys.argv[10]


#############################################################################################################
#############################################################################################################


root_dir=files_path
os.chdir(root_dir)

degree_min = -1.0
degree_max = 1.0 

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0 
        ),
        # RandGaussianNoised(
        #    keys=["image"],
        #    prob=0.30,
        # ),

        # RandBiasFieldd(
        #     keys=["image"],
        #     degree=4, 
        #     coeff_range=(0.4, 0.7),
        #     prob=0.40,
        # ),

        # ScaleIntensityd(
        #     keys=["image"], minv=0.0, maxv=1.0
        # ),

        RandAffined(
            keys=["image", "label"],
            rotate_range=[(degree_min,degree_max),(degree_min,degree_max),(degree_min,degree_max)],
            mode=("bilinear", "nearest"),
            padding_mode=("zeros"),
            prob=0.99,
        ),

        # RandAdjustContrastd(
        #     keys=["image"],
        #     gamma=(0.3, 0.6),
        #     prob=0.50,
        # ),

        # ScaleIntensityd(
        #     keys=["image"], minv=0.0, maxv=1.0
        # ),
        ToTensord(keys=["image", "label"]),
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),
        # RandAffined(
        #     keys=["image", "label"],
        #     rotate_range=[(degree_min,degree_max),(degree_min,degree_max),(degree_min,degree_max)],
        #     mode=("bilinear", "nearest"),
        #     padding_mode=("zeros"),
        #     prob=0.99,
        # ),
        ToTensord(keys=["image", "label"]),
    ]
)

run_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),
        ToTensord(keys=["image"]),
    ]
)


#############################################################################################################
#############################################################################################################


print("Loading data ...")

datasets = files_path + json_file

if status_train_proc > 0:

    train_datalist = load_decathlon_datalist(datasets, True, "training")
    train_ds = CacheDataset(
        data=train_datalist, transform=train_transforms,
        cache_num=100, cache_rate=1.0, num_workers=2,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    
    val_datalist = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(
        data=val_datalist, transform=train_transforms,
        cache_num=50, cache_rate=1.0, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
else:

    run_datalist = load_decathlon_datalist(datasets, True, "running")
    run_ds = CacheDataset(
        data=run_datalist, transform=run_transforms,
        cache_num=50, cache_rate=1.0, num_workers=2,
    )
    run_loader = DataLoader(
        run_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

#############################################################################################################
#############################################################################################################


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#device = torch.device('cpu')
#map_location = torch.device('cpu')


print("Defining the model ...")

# model = AttentionUnet(spatial_dims=3,
#                      in_channels=1,
#                      out_channels=cl_num+1,
#                      channels=(32, 64, 128, 256, 512),
#                      strides=(2,2,2,2),
#                      kernel_size=3,
#                      up_kernel_size=3,
#                      dropout=0.5).to(device)


# model = AttentionUnet(spatial_dims=3,
#                      in_channels=1,
#                      out_channels=cl_num+1,
#                      channels=(16, 32, 64, 128, 256),
#                      strides=(2,2,2,2),
#                      kernel_size=3,
#                      up_kernel_size=3,
#                      dropout=0.5).to(device)

model = UNet(spatial_dims=3,
    in_channels=1,
    out_channels=cl_num+1,
    channels=(32, 64, 128, 256, 512),
    strides=(2,2,2,2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=1,
    act='PRELU',
    norm='INSTANCE',
    dropout=0.5
).to(device)


#############################################################################################################
#############################################################################################################

if status_load_check > 0 :
    print("Loading the checkpoint ...")
    model.load_state_dict(torch.load(os.path.join(check_path, (roi_type+"_best_metric_model.pth"))), strict=False)
    model.eval()
    

#############################################################################################################
#############################################################################################################

if status_train_proc > 0:

    print("Training ...")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (res, res, res), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
                )
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(check_path, (roi_type+"_best_metric_model.pth"))
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    torch.save(
                        model.state_dict(), os.path.join(check_path, (roi_type+"_latest_metric_model.pth"))
                    )
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1
        return global_step, dice_val_best, global_step_best


    #############################################################################################################
    #############################################################################################################


    # max_iterations = 20000
    eval_num = 200

    # post_label = AsDiscrete(to_onehot=True, num_classes=cl_num+1)
    # post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=cl_num+1)
    # dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    post_label = AsDiscrete(to_onehot=cl_num+1)
    post_pred = AsDiscrete(argmax=True, to_onehot=cl_num+1)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []




    #############################################################################################################
    #############################################################################################################


    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )


    #############################################################################################################
    #############################################################################################################

    print("Generating validation labels ... ")

    for x in range(len(val_datalist)):

        case_num = x
        img_name = val_datalist[case_num]["image"]
        # case_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        case_name = os.path.split(img_name)[1]
        out_name = results_path + "/cnn-lab-" + case_name

        print(case_num, out_name)

        img_tmp_info = nib.load(img_name)

        with torch.no_grad():
            # img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
            img = val_ds[case_num]["image"]
            run_inputs = torch.unsqueeze(img, 1).cuda()
            run_outputs = sliding_window_inference(
                run_inputs, (res, res, res), 4, model, overlap=0.8
            )

            out_label = torch.argmax(run_outputs, dim=1).detach().cpu()[0, :, :, :]
            out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
            nib.save(out_lab_nii, out_name)

else: 

    #############################################################################################################
    #############################################################################################################

    print("Running ...")

    for x in range(len(run_datalist)):

        case_num = x
        img_name = run_datalist[case_num]["image"]
        # case_name = os.path.split(run_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        case_name = os.path.split(img_name)[1]
        out_name = results_path + "/cnn-lab-" + case_name

        print(case_num, out_name)

        img_tmp_info = nib.load(img_name)

        with torch.no_grad():
            # img_name = os.path.split(run_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
            img = run_ds[case_num]["image"]
            run_inputs = torch.unsqueeze(img, 1).cuda()
            run_outputs = sliding_window_inference(
                run_inputs, (res, res, res), 4, model, overlap=0.8
            )

            out_label = torch.argmax(run_outputs, dim=1).detach().cpu()[0, :, :, :]
            out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
            nib.save(out_lab_nii, out_name)


#############################################################################################################
#############################################################################################################




#############################################################################################################
#############################################################################################################





