# Qi Zhang, Zhouhang Luo, Tao Yu, and Hui Huang. View Transformation Robustness for Multi-View 3D Object Reconstruction with Reconstruction Error-Guided View Selection. AAAI 2025

This is the official implementation for the AAAI 2025 paper `VTR'.

![VTR](https://github.com/zqyq/VTR/blob/main/pipeline.png)

## Abstract
View transformation robustness (VTR) is critical for deeplearning-based multi-view 3D object reconstruction models, which indicates the methods’ stability under inputs with various view transformations. However, existing research seldom focuses on view transformation robustness in multi-view 3D object reconstruction. One direct way to improve the models’ VTR is to produce data with more view transformations and add them to model training. Recent progress on large vi- sion models, particularly Stable Diffusion models, has provided great potential for generating 3D models or synthesizing novel view images with only a single image input. To fully utilize the power of Stable Diffusion models without causing extra inference computation burdens, we propose to generate novel views with Stable Diffusion models for better view transformation robustness. Instead of synthesizing random views, we propose a reconstruction error-guided view selection method, which considers the reconstruction errors’ spatial distribution of the 3D predictions and chooses the views that could cover the reconstruction errors as much as possible. The methods are trained and tested on sets with large view transformations to validate the 3D reconstruction models’ robustness to view transformations. Extensive experiments demonstrate that the proposed method can outperform state-of-the-art 3D reconstruction methods and other view transformation robustness comparison methods.

## Preliminary work
We should do some preliminary work before install dependencies.

```
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

## Dataset
You can download the dataset at this [link](https://pan.baidu.com/s/1-7ELT53x0M7pt8IG93adcg?pwd=cagb) code: cagb 
We have ShapeNetRendering dataset and ShapeNetVox32 dataset. ShapeNetRendering dataset has images of object, ShapeNetVox32 dataset has the voxel representation of object.

## Dependencies
- python==3.9
- pytorch & torchvision
- tensorboardX
- numpy
- matplotlib
- pillow
- opencv-python
- tqdm
- argparse
- shutil
- tqdm
- rich
- pix2vox++, following their [instruction](https://gitlab.com/hzxie/Pix2Vox)
- LRGT, following their [instruction](https://github.com/LiyingCV/Long-Range-Grouping-Transformer)
- Zero123, following their [instruction](https://github.com/cvlab-columbia/zero123/tree/main)
You can also use command to install requirements.txt

```
pip install -r requirements.txt
```


## Training
For training, you first pretrain the reconstruction model, pix2vox++ and LRGT (the pretrained models are as follow). Then you can simply use the following command: `sh train_ours.sh`

## Testing
We provide the testing script, which you can run as follows: `sh test_on_voxel.sh`

## Pretrained models
You can download the checkpoints at this [link](https://pan.baidu.com/s/1LSk4DAqIp9aqxlILJ0kUaQ?pwd=r9s3) code: r9s3

## Acknowledgement
This work was supported in parts by NSFC (62202312, U21B2023), Guangdong Basic and Applied Basic Research Foundation (2023B1515120026), Shenzhen Science and Technology Program (KQTD 20210811090044003, RCJC20200714114435012), and Scientific Development Funds from Shenzhen University.

## Reference
If you feel the paper is useful or use the code of the paper, please cite our paper:
```
@inproceedings{zhang2025view,
  title={View Transformation Robustness for Multi-View 3D Object Reconstruction with Reconstruction Error-Guided View Selection},
  author={Zhang, Qi and Luo, Zhouhang and Yu, Tao and Huang, Hui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10076--10084},
  year={2025}
}
```
