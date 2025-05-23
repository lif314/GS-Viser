# Offline Viser Viewer for Vanilla 2D and 3D Gaussian Splatting

<p align="center">
  <a href="">
    <img src="./assets/gs-viser-test.gif" alt="teaser" width="95%">
  </a>
</p>

## Installation

```bash
git clone https://github.com/lif314/GS-Viser.git --recursive
cd GS-Viser
pip install -r requirements.txt

# Install the 2D/3D GS environment, including torch, etc.
pip install 2d_gaussian_splatting/submodules/diff-surfel-rasterization
pip install 2d_gaussian_splatting/submodules/simple-knn
pip install 3d_gaussian_splatting/submodules/diff-gaussian-rasterization
pip install 3d_gaussian_splatting/submodules/fused-ssim
```

## Usage
- Linux
```bash
# 2D Gaussian Splatting
gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
# 3D Gaussian Splatting
gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
```

- Windows PowerShell
```bash
# 2D Gaussian Splatting
$env:gs_mode = "2d"; python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
# 3D Gaussian Splatting
$env:gs_mode = "3d"; python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
```

- Windows CMD
```bash
# 2D Gaussian Splatting
set gs_mode="2d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
# 3D Gaussian Splatting
set gs_mode="3d" python gs_viewer.py <path to pre-trained model> <or direct path to the ply file> -s <data source path>
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

