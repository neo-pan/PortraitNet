# Reproduce PortraitNet

This is a re-implementation of PortraitNet by Zhang et al.(2019).

## Requirements

Before running the code, please install requirements using pip:
```bash
pip install -r requirements.txt
```

## Data

Before running the code, please download data from [here](https://github.com/dong-x16/PortraitNet#download-datasets), and unzip them into the `data` folder.
Otherwise, you need to modify config files under `configs/data` to change the `data_root`.

## Training

We provide several pre-set experiment configs in `configs/experiment`, to start a training job, simply the following command:
```bash
python train.py experiment=v2-eg-full
```
This command starts to train PortraitNet with MobileNet-v2 backbone, with both two auxiliary losses, on the EG1800 dataset.
And for more details, please refer to other experiment configs.

To set a different cuda device, for example `cuda:2`, use:
```bash
python train.py experiment=v2-eg-full trainer.devices=\[2\]
```

## Evaluating

To evaluate, please set correct checkpoint path, for example:
```bash
python eval.py experiment=v2-eg-full ckpt_path=checkpoints/v2-eg-full-epoch_0499.ckpt
```

## Reference

Zhang, S.-H., Dong, X., Li, H., Li, R., & Yang, Y.-L. (2019). PortraitNet: Real-time portrait segmentation network for mobile device. Computers & Graphics, 80, 104-113. Elsevier.