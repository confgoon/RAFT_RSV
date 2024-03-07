# RAFT_RSV
This repository contains the source code for our paper:

[Optical Flow-Based Waterway Velocity Detection Algorithm Under Complex Illumination Conditions]

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft_rsv
conda activate raft_rsv
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)


## Acknowledgement
Parts of code are adapted from the following repositories. We thank the authors for their great contribution to the community:
- [RAFT](https://github.com/princeton-vl/RAFT)
