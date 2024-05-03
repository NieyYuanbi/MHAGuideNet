# MHAGuideNet: A 3D Pre-trained Guidance Model for Alzheimer's Disease Diagnosis Using 2D Multi-Planar sMRI Images

## Project Overview
This project is created based on the effective network to diagnose the AD within structural magnetic resonance (sMRI) data. It currently includes the main algorithm modules related to the paper, along with the desensitized datasets provided in this repository. To reproduce the results of the paper, adjusting certain network structure parameters according to the specific dataset is all that's needed.


## Data
The dataset used in this study were acquired from ADNI, which is available freely for all researcher and scientist for experiments on Alzheimer's disease and can be easily downloaded from ADNI websites: http://adni.loni.usc.edu. We release the desensitized data used in the project, named `train.pkl`, `val.pkl` and `MCI_data`. You can download from [https://drive.google.com/drive/folders/1QRoL1H2CgCKn9rubYf9LQgFyxmkuVFzA?usp=sharing](https://drive.google.com/drive/folders/1QRoL1H2CgCKn9rubYf9LQgFyxmkuVFzA?usp=sharing). The data should be processed to `.nii` before used.


## Reproducing Paper Results
If you wish to reproduce the results of the paper, follow these steps:
- Create a conda environment with `torch1.10.0` + `torchvision0.11.1`.
- Clone the repository to your local environment:
  ``https://github.com/NieyYuanbi/MHAGuideNet``
- Adjust network structure parameters: Depending on your dataset, you may need to modify the code's network structure parameters to fit your data.
- Run the `train.py`.

## Contact Information
For any questions or suggestions, feel free to reach out.
Email: nieyuanbi@126.com. Thank you for using our MHAGuideNet!

  
