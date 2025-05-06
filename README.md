# Offine Viser Viewer for 2D and 3D Gaussian Splatting

## Installation
- python version: <= 3.8

```bash
cd GS-Viser-Viewer
pip install viser==0.1.29
pip install lightning==1.8.4
```

## Usage
- Linux
```bash
# 2D or 3D Gaussian Splatting
gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>

### enable transform mode
gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --enable_transform
gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --enable_transform
```

- Windows
```bash
# 2D or 3D Gaussian Splatting
set gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
set gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>

### enable transform mode
set gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --enable_transform
set gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path> --enable_transform
```

### Control 
- **'q/e'** for up & down
- **'w/a/s/d'** for moving
- Mouse wheel for zoom in/out

## Acknowledgements
This project is built upon the following works
- [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting)
- [Original 3D GS Github](https://github.com/graphdeco-inria/gaussian-splatting)
- [Viser](https://github.com/nerfstudio-project/viser)
- [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning)
- [2D GS Viser Viewer](https://github.com/hwanhuh/2D-GS-Viser-Viewer)

