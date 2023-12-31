# MFGF-UNet
The code of MFGF-UNet is from [A Multi-modality Fusion and Gated Multi-filter U-Net for Water Area Segmentation of Remote Sensing Images](https://www.preprints.org/manuscript/202312.1143)

The code of MFGF-UNet can be used for academic research only, please do not use them for commercial purposes. If you have any problem with the code, please contact: rfwang@xidian.edu.cn or zhg_chenchen@163.com.

## 1. Prepare data

Data Preparing

The dataset directory structure of the whole project is as follows:

```python
├── MFGF-UNet
│   └──...
└── data
    └──WIPI
        └──SAR
            │   ├── 0.tiff
            │   ├── 1.tiff
            │   ├── ...
            │   └── *.tiff
        └──WI
            │   ├── 0.tiff
            │   ├── 1.tiff
            │   ├── ...
            │   └── *.tiff
        └──Labels
            │   ├── 0.tiff
            │   ├── 1.tiff
            │   ├── ...
            │   └── *.tiff
```
## 2. Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3.Train/Test
Train
python train.py --data_path 'your data path' --root_path 'your workspace' --epochs train epochs -b batch_size 
Test
python CESHI.py
