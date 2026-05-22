# Remote_Sensing
Remote Sensing Imagery Segmentation.
## Project Structure
```
remote_sensing/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── utils/
│   ├── dataset.py               # Dataset loading
│   └── metrics.py               # IoU, F1, precision/recall
├── phase1_baseline/
│   └── resnet_unet.py           # ResNet-backed U-Net baseline
├── phase2_satmae/
│   └── satmae_encoder.py        # SatMAE++ frozen encoder wrapper
├── phase3_swin_unet/
│   └── swin_unet_decoder.py     # Swin-Unet decoder + full model
├── phase4_sam_pseudolabel/
│   └── sam_pipeline.py          # SAM zero-shot pseudo-labeling
└── train.py                     # Unified training entry point
```

## Dataset
Uses WHU Building Dataset (aerial imagery, ~180k building instances).
Download: http://gpcv.whu.edu.cn/data/building_dataset.html

## Phase Summary
- **Phase 1**: ResNet34 U-Net baseline → establishes IoU benchmark
- **Phase 2**: Replace encoder with frozen SatMAE++ ViT
- **Phase 3**: Add Swin Transformer decoder for hierarchical attention
- **Phase 4**: SAM pseudo-labeling to expand unlabeled data



## USEFUL INSTRUCTIONS
- For **phase 4** run this command in your terminal
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O weights/sam_vit_h.pth
-Segment Anything Model **SAM**
  pip install git+https://github.com/facebookresearch/segment-anything.git  
