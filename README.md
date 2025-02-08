# Qi Zhang, Zhouhang Luo, Tao Yu, and Hui Huang. View Transformation Robustness for Multi-View 3D Object Reconstruction with Reconstruction Error-Guided View Selection. AAAI 2025

This is the official implementation for the AAAI 2025 paper `VTR'.



## Abstract
View transformation robustness (VTR) is critical for deeplearning-based multi-view 3D object reconstruction models, which indicates the methods’ stability under inputs with various view transformations. However, existing research seldom focuses on view transformation robustness in multi-view 3D object reconstruction. One direct way to improve the models’ VTR is to produce data with more view transformations and add them to model training. Recent progress on large vi- sion models, particularly Stable Diffusion models, has provided great potential for generating 3D models or synthesizing novel view images with only a single image input. To fully utilize the power of Stable Diffusion models without causing extra inference computation burdens, we propose to generate novel views with Stable Diffusion models for better view transformation robustness. Instead of synthesizing random views, we propose a reconstruction error-guided view selection method, which considers the reconstruction errors’ spatial distribution of the 3D predictions and chooses the views that could cover the reconstruction errors as much as possible. The methods are trained and tested on sets with large view transformations to validate the 3D reconstruction models’ robustness to view transformations. Extensive experiments demonstrate that the proposed method can outperform state-of-the-art 3D reconstruction methods and other view transformation robustness comparison methods.


## Dependencies
- python
- pytorch & torchvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- tqdm
- argparse
- shutil


## Training

## Testing

## Pretrained models

## Acknowledgement
This work was supported in parts by NSFC (62202312, U21B2023), Guangdong Basic and Applied Basic Research Foundation (2023B1515120026), Shenzhen Science and Technology Program (KQTD 20210811090044003, RCJC20200714114435012), and Scientific Development Funds from Shenzhen University.

## Reference
```
@inproceedings{VTR25,
title={View Transformation Robustness for Multi-View 3D Object Reconstruction With Reconstruction Error-Guided View Selection},
author={Qi Zhang and Zhouhang Luo and Tao Yu and Hui Huang},
booktitle={AAAI},
pages={},
year={2025},
}
```
