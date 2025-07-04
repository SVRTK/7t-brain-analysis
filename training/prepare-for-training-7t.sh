#!/bin/sh


all=(


neo-032_ses-01_7t_recon
neo-032_ses-02_7t_recon
neo-035_ses-01_7t_recon
neo-038_ses-01_7t_recon
neo-039_ses-01_7t_recon
neo-040_ses-01_7t_recon
neo-044_ses-01_7t_recon
neo-045_ses-01_7t_recon
neo-046_ses-01_7t_recon
neo-047_ses-01_7t_recon
neo-048_ses-01_7t_recon
neo-049_ses-01_7t_recon
neo-050_ses-01_7t_recon
neo-051_ses-01_7t_recon
neo-053_ses-01_7t_recon
neo-059_ses-02_7t_recon
neo-064_ses-01_7t_recon
neo-066_ses-01_7t_recon
neo-068_ses-01_7t_recon
neo-071_ses-01_7t_recon
neo-072_ses-01_7t_recon
neo-073_ses-01_7t_recon
neo-074_ses-01_7t_recon
neo-077_ses-01_7t_recon


)



src=/data2/scratch/training/bounti-7t
mirtk=/software/MIRTK/build/lib/tools





in=org-datasets

out=training-aff-7t-072025

cnn=results-01042025

mkdir tmp

for ((i=0;i<${#all[@]};i=i+3));
do

    s=${all[$i]}

    echo  ${i} / ${#all[@]} : ${s}


#    remove negative values from the image

    ${mirtk}/nan ${in}/${s}.nii.gz 1000000
    
#    extract left / right label for reorientation

    ${mirtk}/extract-label-brain-multi-bounti-lr  ${in}/cnn-lab-${s}.nii.gz ${in}/lr-lab-${s}.nii.gz
    
#    copy reference space origin from the recon origin

    ${mirtk}/edit-image ref/ref-t2.nii.gz tmp/tmp-ref.nii.gz -copy-origin ${in}/${s}.nii.gz

    ${mirtk}/edit-image ref/lr-lab.nii.gz tmp/tmp-lr-ref.nii.gz -copy-origin ${in}/${s}.nii.gz
    
#    register subject to the reference space

    ${mirtk}/register tmp/tmp-lr-ref.nii.gz ${in}/lr-lab-${s}.nii.gz -model Affine -dofin i.dof  -dofout dofs/aff-${s}.dof -v 0


#   crop the image and run n4 bias correction
    
    ${mirtk}/threshold-image ${in}/${s}.nii.gz tmp/m.nii.gz 1.0
    
    ${mirtk}/extract-connected-components tmp/m.nii.gz tmp/m.nii.gz

    ${mirtk}/crop-image ${in}/${s}.nii.gz tmp/m.nii.gz tmp/crop-${s}.nii.gz
    
    /software/N4BiasFieldCorrection -i tmp/crop-${s}.nii.gz -o tmp/n4-crop-${s}.nii.gz > tmp/t.txt
    
#    transform to the reference space
    
    transform-image ${in}/${s}.nii.gz ${out}/t2-${s}.nii.gz -target tmp/tmp-ref.nii.gz -dofin dofs/aff-${s}.dof

    transform-image ${in}/cnn-lab-${s}.nii.gz ${out}/multi-lab-${s}.nii.gz -target tmp/tmp-ref.nii.gz -dofin dofs/aff-${s}.dof -labels
    
    convert-image ${out}/t2-${s}.nii.gz ${out}/t2-${s}.nii.gz -rescale 0 5000 -short


#   try to visualise the image

    /software/itksnap-bin/bin/itksnap-g ${out}/t2-${s}.nii.gz -s ${out}/multi-lab-${s}.nii.gz  > nul 2>&1
    
    
#    print input record for the .json file

     echo "{"
     echo %image%: %${out}/t2-${s}.nii.gz%,
     echo %label%: %${out}/multi-lab-${s}.nii.gz%
     echo "},"



done



 
