
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
    # AddChanneld,
    Compose,
    CropForegroundd,
    RandSpatialCropd, 
    EnsureChannelFirstd,
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
    Flip
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


from monai.metrics import DiceMetric, ConfusionMatrixMetric, compute_confusion_matrix_metric
from monai.networks.nets import UNet, AttentionUnet
from monai.data import (
    DataLoader,
    CacheDataset, decollate_batch)

import torch
import warnings
import torchvision

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
#to_tensor = ToTensor()
#to_numpy = ToNumpy()


############## DEFINE TRAIN / TEST SETTINGS ##############

#print(1, sys.argv[1])
#print(2, sys.argv[2])
#print(3, sys.argv[3])
#print(4, sys.argv[4])
#print(5, sys.argv[5])
#print(6, sys.argv[6])
#print(7, sys.argv[7])
#print(8, sys.argv[8])
#print(9, sys.argv[9])
#print(10, sys.argv[10])
#print(11, sys.argv[11])
#print(12, sys.argv[12])


files_path = sys.argv[1]
check_path = sys.argv[2]
json_file = sys.argv[3]
results_path = sys.argv[4]

res = int(sys.argv[5]) # patch-size used: 128x128x128
cl_num = int(sys.argv[6]) # label number: 25

status_train_proc = int(sys.argv[7]) # if I want to train or not (1) : use 1 to train, 0 to test

status_load_check = int(sys.argv[8]) # if I want to load checkpoint or not (1): use 0 for trainnig from scratch or 1 to load checkpoint
max_iterations = int(sys.argv[9]) # iterations for training (e.g: 200 000)
roi_type = sys.argv[10]


#print(1, files_path)
#print(2, check_path)
#print(3, json_file)
#print(4, results_path)
#print(5, check_path)
#print(6, results_path)
#print(7, res)
#print(8, cl_num)
#print(9, status_train_proc)
#print(10, status_load_check)
#print(11, max_iterations)
#print(12, roi_type)


############## DEFINE TRAIN / TEST TRANSFORMATIONS ##############
# ROTATION
degree_min = -0.1
degree_max = 0.1

# BIAS-FIELD
coef_min = 0.07
coef_max = 0.3
#

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),

        RandBiasFieldd(
            keys=["image"],
            degree=3,
            coeff_range=(coef_min, coef_max),  
            prob=0.20,
        ),

        ScaleIntensityd(
            keys=["image"], minv=0.0, maxv=1.0
        ),

        RandSpatialCropd(keys= ["image", "label"], roi_size= (128,128,128), random_center = True, random_size = False),
    #    RandGaussianNoised(
    #        keys=["image"],
    #        prob=0.30,
    #    ),

        # RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 4.5)
        # ),

#        RandFlipd(keys=["image", "label"], prob=0.20, spatial_axis=1
#        ),


        RandAffined(
            keys=["image", "label"],
            rotate_range=[(degree_min,degree_max),(degree_min,degree_max),(degree_min,degree_max)],
            mode=("bilinear", "nearest"),
            padding_mode=("zeros"), 
            prob=0.001, 
        ),

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

############## DEFINE TRAIN & TEST DATASETS  ##############

#image_paths_train = sorted(glob.glob(os.path.join(train_dataset, "*.nii.gz")))
#label_paths_train = sorted(glob.glob(os.path.join(train_labels, "*.nii.gz")))
#
#image_paths_test = sorted(glob.glob(os.path.join(test_dataset, "*.nii.gz")))
#label_paths_test = sorted(glob.glob(os.path.join(test_labels, "*.nii.gz")))
#
#dict_data_train_val = [{"image": image_scan, "label": labels_seg}
#         for image_scan, labels_seg in zip(image_paths_train, label_paths_train)]
#
#dict_test = [{"image": image_scan}
#        for image_scan in zip(image_paths_test)]
#
#dic_train, dic_val = train_test_split(dict_data_train_val, train_size = 0.9, random_state = 1, shuffle= 'False')


#############################################################################################################
# LOADING DATASETS
#############################################################################################################
print("Loading data ...")

#if status_train_proc > 0:
#
#    train_ds = CacheDataset(
#        data = dic_train,
#        transform = train_transforms,
#        cache_num = 100,
#        cache_rate = 1.0,
#        num_workers = 8)
#
#    train_loader = DataLoader(
#    train_ds, batch_size= 2, shuffle=True, num_workers=8, pin_memory=True)
#
#    val_ds = CacheDataset(
#        data = dic_val,
#        transform = val_transforms,
#        cache_num = 6,
#        cache_rate = 1.0,
#        num_workers = 4)
#
#    val_loader = DataLoader(
#    val_ds, batch_size= 1, shuffle=False, num_workers=4, pin_memory=True)
#
#else:
#
#    run_ds = CacheDataset(
#        data = dict_test,
#        transform= run_transforms,
#        cache_num = 6,
#        cache_rate = 1.0,
#        num_workers = 4)
#
#    run_loader = DataLoader(
#        run_ds, batch_size= 1, shuffle=False, num_workers=4, pin_memory=True)



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
# DEFINING MODEL
#############################################################################################################
print("Defining the model ...")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import torch
# print("CUDA Available:", torch.cuda.is_available())  
# print("CUDA Device Count:", torch.cuda.device_count())


# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#model = AttentionUnet(spatial_dims=3,
#                     in_channels=1,
#                     out_channels=cl_num+1,
#                     channels=(16, 32, 64, 128, 256),
#                     strides=(2,2,2,2),
#                     kernel_size=3,
#                     up_kernel_size=3,
#                     dropout=0.5).to(device)


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
# USE CHECKPOINT TO CONTINUE TRAINING OR LOAD PRE-TRAINED MODEL BOUNTI TO INCLUDE IN TRAINING
#############################################################################################################

if status_load_check > 0 :

    print("Loading BOUNTI checkpoints into the model for training ...")
    # to continue running the model
    model.load_state_dict(torch.load(os.path.join(check_path, (roi_type+"_best_metric_model.pth"))), strict=False) # before was roi_type + "_best_metric_model.pth"
    model.eval()

# # If want to pretrain network with BOUNTI (19 tissues):
#     pre_model = AttentionUnet(spatial_dims=3,
#                      in_channels=1,
#                      out_channels=20,
#                      channels=(16, 32, 64, 128, 256),
#                      strides=(2,2,2,2),
#                      kernel_size=3,
#                      up_kernel_size=3,
#                      dropout=0.5).to(device)
    
#     pre_model.load_state_dict(torch.load(os.path.join(check_path, ("pre_best_metric_model.pth"))), strict=False) # before was roi_type + "_best_metric_model.pth"

#     print("Loading the checkpoint ...")
#     # load only necessary layers
#     pretrained_dict = pre_model.state_dict()
#     new_dict_model = model.state_dict()
    
#     processed_dict = {}

#     for k in new_dict_model.keys():
#         decomposed_key = k.split(".")
#         if ("pre_model" in decomposed_key):
#             pretrained_key = ".".join(decomposed_key[1:])
#             processed_dict[k] = pretrained_dict[pretrained_key]

#     model.load_state_dict(processed_dict, strict=False)
    
#############################################################################################################
# TRAINING
#############################################################################################################

if status_train_proc > 0:

    print("Training ...")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def validation(epoch_iterator_val):
        model.eval()
#        writer_val = SummaryWriter(os.path.join(results_path, 'run/validation'))
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
        
        # Validation Loss
#        writer_val.add_scalar("Validation-DICE-Loss-Mean", mean_dice_val.item() , global_step)

        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
#        writer_train = SummaryWriter(os.path.join(results_path, 'run/training'))
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
            
            ###### PLOTING TRAINING AND VALIDATION LOSSES in TENSORBOARD ######
#            writer_train.add_scalar("Training-DICE-Loss-Epoch", loss.item(), global_step)

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
    # DEFINE PARAMETERS FOR TRAINING, etc.
    #############################################################################################################
    eval_num = 200
    post_label = AsDiscrete(to_onehot=cl_num+1)
    post_pred = AsDiscrete(argmax=True, to_onehot=cl_num+1)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    #############################################################################################################
    # START TRAINING/TESTING PROCESS
    #############################################################################################################

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )

else:

    #############################################################################################################
    # FOR TESTING MODEL
    #############################################################################################################

    print("Running ...")
    
    
    flp_run = Flip(1)

    def replace_dhcp(fl_val_outputs):
    
        org_fl_val_outputs = fl_val_outputs.clone();

        i_org = 1 ; i_fl = 2 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 2 ; i_fl = 1 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 3 ; i_fl = 4 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 4 ; i_fl = 3 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 5 ; i_fl = 6 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 6 ; i_fl = 5 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 7 ; i_fl = 8 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 8 ; i_fl = 7 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 9 ; i_fl = 10 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 10 ; i_fl = 9 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 11 ; i_fl = 12 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 12 ; i_fl = 11 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 13 ; i_fl = 14 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 14 ; i_fl = 13 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 15 ; i_fl = 16 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 16 ; i_fl = 15 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 17 ; i_fl = 18 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 18 ; i_fl = 17 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 19 ; i_fl = 20 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 20 ; i_fl = 19 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 21 ; i_fl = 22 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 22 ; i_fl = 21 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 23 ; i_fl = 24 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 24 ; i_fl = 23 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 25 ; i_fl = 25 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 26 ; i_fl = 26 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 27 ; i_fl = 27 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 28 ; i_fl = 29 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 29 ; i_fl = 28 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 30 ; i_fl = 30 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 31 ; i_fl = 32 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 32 ; i_fl = 31 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        
        i_org = 33 ; i_fl = 34 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 34 ; i_fl = 33 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 35 ; i_fl = 36 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 36 ; i_fl = 35 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 37 ; i_fl = 38 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 38 ; i_fl = 37 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 39 ; i_fl = 40 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 40 ; i_fl = 39 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;

        i_org = 41 ; i_fl = 41 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 42 ; i_fl = 42 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;
        i_org = 43 ; i_fl = 43 ; org_fl_val_outputs[0,i_org,:,:,:] = fl_val_outputs[0,i_fl,:,:,:] ;


        return org_fl_val_outputs
    
    
    
    
    
    model.load_state_dict(torch.load(os.path.join(check_path, (roi_type + "_best_metric_model.pth"))), strict=False)
    model.eval()
 # load checkpoint
    mean_dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)

    for x in range(len(run_datalist)):

        case_num = x
        img_name = run_datalist[case_num]["image"]
        print(img_name)
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
            
            fl_run_inputs = torch.unsqueeze(img, 1).cuda()
            fl_run_inputs = flp_run(fl_run_inputs)
            fl_run_outputs = sliding_window_inference(
                fl_run_inputs, (res, res, res), 4, model, overlap=0.8
            )
            fl_run_outputs_tmp = flp_run(fl_run_outputs.clone())
            fl_run_outputs_fin = replace_dhcp(fl_run_outputs_tmp.clone())

            sum_run_outputs = (run_outputs.clone() + fl_run_outputs_fin.clone()) / 2.0

            out_label = torch.argmax(sum_run_outputs, dim=1).detach().cpu()[0, :, :, :]
            out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
            nib.save(out_lab_nii, out_name)


#############################################################################################################
