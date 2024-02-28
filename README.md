# RelightableAvatar (AAAI'2024)

### [Project Page](https://wenbin-lin.github.io/RelightableAvatar-page/) | [Paper](https://arxiv.org/abs/2312.12877) | [Video](https://www.youtube.com/watch?v=asxefE2Ey6E) 

## Installation

**Environment Setup**

This repository has been tested on the Python 3.8, Pytorch 1.10.1 with CUDA 11.3, Ubuntu 22.04.
We use a GTX 3090 for training and inference, please make sure enough GPU memory if using other cards.
```
conda env create -f environment.yml
conda activate RAvatar
```
It is recommended to build pytorch3d from source.
```
wget -O pytorch3d-0.4.0.zip https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.zip
unzip pytorch3d-0.4.0.zip
cd pytorch3d-0.4.0 && python setup.py install && cd ..
```

**SMPL Setup**

Download smpl model from [SMPL website](https://smpl.is.tue.mpg.de/), we use the neutral model from SMPL_python_v.1.1.0, and put the model files (`basicmodel_f_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` and `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`) to `$ROOT/data/smplx/smpl/`.

## Test with Trained Models

- Download trained models from [Google Drive](https://drive.google.com/drive/folders/1NKGitjhAwHZT_3KJmXQo_SnvfycW6865), and put them to `$ROOT/data/trained_model/`.
- Render subjects in novel light and novel poses.
```
python run_material.py --type visualize --cfg_file configs/material_ps_m3c.yaml exp_name material_ps_m3c novel_light True vis_pose_sequence True
```
Results are saved in `$ROOT/data/`. Target environment light and pose sequence can be set by 'novel_light_path' and 'novel_poses_path' parameter in the configuration.

## Train from Scratch

### Dataset preparation

- For the People-Snapshot dataset
    1. Download the People-Snapshot dataset [here](https://graphics.tu-bs.de/people-snapshot).
    2. Create a soft link by: `ln -s /path/to/people_snapshot ./data/people_snapshot`
    3. Run this script to process the dataset: `python process_snapshot.py`

- For the ZJU-Mocap dataset, we follow [AnimatableNeRF](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md) for dataset preparation. Then create a soft link by: `ln -s /path/to/zju_mocap ./data/zju_mocap`

### Train the model in 3 stages

**1. Geometry and Motion Reconstruction**
Training.
```
python train_geometry.py --cfg_file configs/geometry_ps_m3c.yaml exp_name geometry_ps_m3c
```
Visualize results of the first stage.
```
python run_geometry.py --type visualize --cfg_file configs/geometry_ps_m3c.yaml exp_name geometry_ps_m3c
```

**2. Light Visibility Estimation**
Generate training data.
```
python run_geometry.py --type visualize --cfg_file configs/geometry_ps_m3c.yaml exp_name geometry_ps_m3c gen_lvis_mesh True
```
Train light visibility model.
```
python train_lvis.py --cfg_file configs/geometry_ps_m3c.yaml exp_name lvis_ps_m3c exp_name_geo geometry_ps_m3c
```

**3. Material and Lighting**
Training.
```
python train_material.py --cfg_file configs/material_ps_m3c.yaml exp_name material_ps_m3c
```
Visualize results of the last stage, relighting with the reconstructed light.
```
python run_material.py --type visualize --cfg_file configs/material_ps_m3c.yaml exp_name material_ps_m3c
```

## TODO

- More datasets (Human3.6M, DeepCap and our synthetic dataset) and pretrained models.
- Release the synthetic dataset.

## Citation

If you find our work useful in your research, please consider citing:

    @article{lin2023relightable,
        title={Relightable and Animatable Neural Avatars from Videos},
        author={Lin, Wenbin and Zheng, Chengwei and Yong, Jun-Hai and Xu, Feng},
        journal={arXiv preprint arXiv:2312.12877},
        year={2023}
    }

Acknowledgements: This repository is built on top of the [AnimableNeRF](https://github.com/zju3dv/animatable_nerf/) codebase, and part of our code is inherited from [InvRender](https://github.com/zju3dv/invrender). We are grateful to the authors for releasing their code.


