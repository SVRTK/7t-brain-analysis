Scripts for automated 7t neonatal brain analysis 
====================

This repository contains the pipelines for [MONAI](https://github.com/Project-MONAI/MONAI)-based automated analysis for 7T neonatal brain MRI executed within [SVRTK dockers](https://hub.docker.com/r/fetalsvrtk/svrtk).


- The repository and code for deep learning segmentation were designed and created by King's College London.

  
- Please email alena.uus (at) kcl.ac.uk if in case of any questions.


<img src="info/auto-svrtk-072025.jpg" alt="AUTOSVRTKEXAMPLE" height="700" align ="center" />


Development of these analysis tools was supported by projects led by Prof Mary Rutherford, Prof Tomoki Arichi, Prof Jonathan O’Muircheartaigh, Prof Shaihan Malik and Prof Jo Hajnal.



Auto processing 
---------------


**The automated docker tags are _fetalsvrtk/svrtk:7t_brain_analysis_amd_ OR _fetalsvrtk/svrtk:7t_brain_analysis_arm_ (M1 chip)**


**AUTOMATED 3D T2w BRAIN SEGMENTATION:**


<img src="info/svrtk-labels.jpg" alt="AUTOSVRTKEXAMPLE" height="200" align ="center" />

*Input data requirements:*
- sufficient SNR and image quality, no extreme shading artifacts
- good quality 3D SVR 
- full ROI coverage
- standard radiological space
- 25-45 weeks PMA
- no extreme structural anomalies
- 7T / 3T

_Note: please use 7t_brain_analysis_arm tag for M1 Mac computers._

```bash

docker pull fetalsvrtk/svrtk:7t_brain_analysis

#auto internal capsule segmentation
docker run --rm  --mount type=bind,source=LOCATION_ON_YOUR_MACHINE,target=/home/data  fetalsvrtk/svrtk:7t_brain_analysis_amd sh -c ' bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh /home/data/your_folder_with_brain_svr_t2_files  /home/data/output_folder_for_segmentations ; '


```




License
-------

The 7t-brain-analysis code and all scripts are distributed under the terms of the
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation version 3 of the License. 

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.


Citation and acknowledgements
-----------------------------

In case you found this repository useful please give appropriate credit to the software.


** Internal capsule segmentation:**
> Chiara Casella, Alena Uus, Luke Dedominicis, Jucha Willers Moore, Benjamin Clayden, Emil Galanides, Philippa Bridgen, Pierluigi Di Cio, Ines Tomazinho, Cidalia Da Costa, Dario Gallo, Sophie Arulkumaran, Maria Deprez, Serena J. Counsell, Joseph V. Hajnal, Jonathan O’Muircheartaigh, Mary A. Rutherford, Shaihan Malik, Tomoki Arichi. (2026) Automated assessment of neonatal internal capsule maturation on T2-weighted MRI
across 7T and 3T. MedrXiv; doi: https://doi.org/****



Disclaimer
-------

This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.

