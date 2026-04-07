#!/usr/bin/env bash -l


src=/home/7t-brain-analysis
mirtk_path=/bin/MIRTK/build/lib/tools


org_t2=$1
proc=$2
out_bounti_lab=$3
out_wm_lab=$4


mode=gpu


if [[ $# -ne 4 ]] ; then

    echo
    echo "------------------------------------------------------------"
    echo
    echo "Usage: please use the following format ..."
    echo "bash /home/7t-brain-analysis/scripts/run-7t-neo-brain-segmentation-ic-multi-bounti-042026.sh [path_to_input_t2w_recon.nii.gz] [path_to_folder_for_tmp_processing] [path_to_output_label.nii.gz] [path_to_wm_label.nii.gz]"
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


${mirtk_path}/convert-image ${org_t2} ${proc}/org-t2.nii.gz


# remove negative

${mirtk_path}/nan ${proc}/org-t2.nii.gz 100000

# crop background

${mirtk_path}/threshold-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz 1.0 > ${proc}/t.txt

${mirtk_path}/crop-image ${proc}/org-t2.nii.gz ${proc}/m-t2.nii.gz ${proc}/crop-t2-128.nii.gz

# run bias correction

${src}/bin/N4BiasFieldCorrection -i ${proc}/crop-t2-128.nii.gz -o ${proc}/n4-crop-t2-128.nii.gz > ${proc}/t.txt


# run reorientation to the standard space

# network extracts l/r labels

${mirtk_path}/edit-image ${src}/templates/ref-lr-global/large-ref-t2-1mm.nii.gz ${proc}/tmp-global-ref.nii.gz -copy-origin ${proc}/crop-t2-128.nii.gz

${mirtk_path}/transform-image ${proc}/n4-crop-t2-128.nii.gz ${proc}/tr-n4-crop-t2-128.nii.gz -target ${proc}/tmp-global-ref.nii.gz 

${mirtk_path}/crop-image ${proc}/tr-n4-crop-t2-128.nii.gz ${proc}/m-t2.nii.gz ${proc}/tr-n4-crop-t2-128.nii.gz 

${mirtk_path}/nan ${proc}/tr-n4-crop-t2-128.nii.gz 100000

${mirtk_path}/pad-3d ${proc}/tr-n4-crop-t2-128.nii.gz ${proc}/pad-crop-t2-128.nii.gz 128 1

# unset PYTHONPATH ; 

weights_lr=${src}/models/atunet_lr_brain_7t_2lab_best_metric_model.pth

python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-${mode}.py 128 2 ${weights_lr} ${proc}/pad-crop-t2-128.nii.gz ${proc}/lr-lab-pad-crop-t2-128.nii.gz


${mirtk_path}/edit-image ${src}/templates/ref-lr-global/ref-t2.nii.gz ${proc}/tmp-ref.nii.gz -copy-origin ${proc}/org-t2.nii.gz

${mirtk_path}/edit-image ${src}/templates/ref-lr-global/lr-lab.nii.gz ${proc}/tmp-lr-ref.nii.gz -copy-origin ${proc}/org-t2.nii.gz

# register to the atlas space

${mirtk_path}/register ${proc}/tmp-lr-ref.nii.gz ${proc}/lr-lab-pad-crop-t2-128.nii.gz -model Affine -dofin ${src}/templates/ref-lr-global/i.dof -dofout ${proc}/aff-d.dof -v 0

${mirtk_path}/transform-image ${proc}/n4-crop-t2-128.nii.gz  ${proc}/tr-n4-t2.nii.gz  -target ${proc}/tmp-lr-ref.nii.gz -dofin ${proc}/aff-d.dof

${mirtk_path}/nan ${proc}/tr-n4-t2.nii.gz  50000


echo 
echo "------------------------------------------------------------"
echo
echo " - RUNNING MULTI-BOUNTI SEGMENTATION ... "
echo
echo "------------------------------------------------------------"
echo
  

# run multi-label network

# unset PYTHONPATH ; 

# 2025
# python ${src}/run_monai_patch_unet_segmentation_1case-2025-gpu.py 128 43 ${src}/patch_unet_new_multi_43_brain_neo_7t_aff_best_metric_model.pth ${proc}/tr-n4-t2.nii.gz ${proc}/multi-lab-tr-t2.nii.gz


# 2026

weights_bounti=${src}/models/patch_atunet_new_multi_43_brain_neo_7t_3t_aff_fix_022026_best_metric_model.pth

python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-${mode}.py 128 43 ${weights_bounti} ${proc}/tr-n4-t2.nii.gz ${proc}/multi-lab-tr-t2.nii.gz

${mirtk_path}/transform-image ${proc}/multi-lab-tr-t2.nii.gz ${proc}/org-space-multi-lab-tr-t2.nii.gz -target ${proc}/org-t2.nii.gz  -dofin_i ${proc}/aff-d.dof -labels

${mirtk_path}/thin-cortex-thick ${proc}/org-space-multi-lab-tr-t2.nii.gz ${out_bounti_lab} > ${proc}/t.txt 



echo 
echo "------------------------------------------------------------"
echo
echo " - RUNNING WM SEGMENTATION ... "
echo
echo "------------------------------------------------------------"
echo
  
  
weights_wm=${src}/models/patch_atunet_ic_dgm_14_brain_neo_7t_3t_aff_best_metric_model.pth

# unset PYTHONPATH ;

python3 ${src}/src/run_monai_patch_atunet_segmentation_1case-2026-flip-14-gpu.py 128 14 ${weights_wm} ${proc}/tr-n4-t2.nii.gz ${proc}/wm-lab-tr-t2.nii.gz

${mirtk_path}/transform-image ${proc}/wm-lab-tr-t2.nii.gz ${out_wm_lab} -target ${proc}/org-t2.nii.gz  -dofin_i ${proc}/aff-d.dof -labels



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

    # rm -r ${proc}/*

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

chmod -R 777 ${proc} ${out_bounti_lab} ${out_wm_lab}



 
