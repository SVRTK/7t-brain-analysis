#!/bin/sh


src=/data2/scratch/training/bounti-7t
mirtk=/software/MIRTK/build/lib/tools


org_t2=$1
proc=$2
out_bounti_lab=$3
out_wm_lab=$4


if [[ $# -ne 4 ]] ; then

    echo
    echo "------------------------------------------------------------"
    echo
    echo "Usage: please use the following format ..."
    echo "bash run-7t-neo-brain-segmentation-2025.sh [path_to_input_t2w_recon.nii.gz] [path_to_folder_for_tmp_processing] [path_to_output_label.nii.gz] [path_to_wm_label.nii.gz]"
    echo
    echo "------------------------------------------------------------"
    echo
    exit

fi 



echo 
echo "------------------------------------------------------------"
echo
echo " - SCRIPT FOR 7T NEONATAL PROCESSING ... "
echo
echo "------------------------------------------------------------"
echo 
  


echo
echo "------------------------------------------------------------"
echo
echo " - input t2 : " ${org_t2}
echo " - processing folder : " ${proc}
echo
echo "------------------------------------------------------------"
echo
echo " - RUNNING PREPROCESSING ... "
echo
echo "------------------------------------------------------------"
echo

if [[ ! -f ${org_t2} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: NO INPUT FILE ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi

if [[ ! -d ${proc} ]];then
    mkdir ${proc}
fi

if [[ ! -d ${proc} ]];then
    echo
    echo "------------------------------------------------------------"
    echo
    echo "ERROR: CANNOT CREATE PROCESSING FOLDER ..."
    echo
    echo "------------------------------------------------------------"
    echo
    exit
fi


${mirtk}/convert-image ${org_t2} ${proc}/org-t2.nii.gz

${mirtk}/nan ${proc}/org-t2.nii.gz 100000

${mirtk}/threshold-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz 1.0 > ${proc}/t.txt

${mirtk}/crop-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-t2-128.nii.gz

/software/N4BiasFieldCorrection -i ${proc}/crop-t2-128.nii.gz -o ${proc}/n4-crop-t2-128.nii.gz > ${proc}/t.txt

${mirtk}/pad-3d ${proc}/n4-crop-t2-128.nii.gz ${proc}/pad-crop-t2-128.nii.gz 128 1

unset PYTHONPATH ; 
python ${src}/run_monai_unet_segmentation_1case-2024.py 128 2 ${src}/unet_lr_brain_7t_2lab_best_metric_model.pth ${proc}/pad-crop-t2-128.nii.gz ${proc}/lr-lab-pad-crop-t2-128.nii.gz

${mirtk}/edit-image ${src}/ref/ref-t2.nii.gz ${proc}/tmp-ref.nii.gz -copy-origin ${proc}/org-t2.nii.gz

${mirtk}/edit-image ${src}/ref/lr-lab.nii.gz ${proc}/tmp-lr-ref.nii.gz -copy-origin ${proc}/org-t2.nii.gz

${mirtk}/register ${proc}/tmp-lr-ref.nii.gz ${proc}/lr-lab-pad-crop-t2-128.nii.gz -model Affine -dofin ${src}/i.dof -dofout ${proc}/aff-d.dof -v 0

${mirtk}/transform-image ${proc}/n4-crop-t2-128.nii.gz  ${proc}/tr-n4-t2.nii.gz  -target ${proc}/tmp-lr-ref.nii.gz -dofin ${proc}/aff-d.dof

${mirtk}/nan ${proc}/tr-n4-t2.nii.gz  50000

echo 
echo "------------------------------------------------------------"
echo
echo " - RUNNING MULTI-BOUNTI SEGMENTATION ... "
echo
echo "------------------------------------------------------------"
echo
  

unset PYTHONPATH ; 
python ${src}/run_monai_patch_unet_segmentation_1case-2025-gpu.py 128 43 ${src}/patch_unet_new_multi_43_brain_neo_7t_aff_best_metric_model.pth ${proc}/tr-n4-t2.nii.gz ${proc}/multi-lab-tr-t2.nii.gz

${mirtk}/transform-image ${proc}/multi-lab-tr-t2.nii.gz ${out_bounti_lab} -target ${proc}/org-t2.nii.gz  -dofin_i ${proc}/aff-d.dof -labels



echo 
echo "------------------------------------------------------------"
echo
echo " - RUNNING WM SEGMENTATION ... "
echo
echo "------------------------------------------------------------"
echo
  

unset PYTHONPATH ; 
python ${src}/run_monai_unet_segmentation_1case-2025-gpu.py 256 20 ${src}/unet_wm_brain_7t_20lab_best_metric_model.pth ${proc}/tr-n4-t2.nii.gz ${proc}/wm-lab-tr-t2.nii.gz

${mirtk}/transform-image ${proc}/wm-lab-tr-t2.nii.gz ${out_wm_lab} -target ${proc}/org-t2.nii.gz  -dofin_i ${proc}/aff-d.dof -labels



# cp ${proc}/multi-lab-org-t2.nii.gz ${proc}/org-t2.nii.gz ${out_bounti_lab}


if [[ ! -f ${out_bounti_lab} ]];then

    echo 
    echo "------------------------------------------------------------"
    echo
    echo "ERROR - LABEL FILE WAS NOT GENERATED ..."
    echo 
    echo "------------------------------------------------------------"
    echo
    exit
    
else

    rm -r ${proc}/*

    echo
    echo "------------------------------------------------------------"
    echo
    echo " - output BOUNTI label : " ${out_bounti_lab}
    echo
    echo " - output WM label : " ${out_wm_lab}
    echo
    echo "------------------------------------------------------------"
    echo


fi


 
