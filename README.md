# DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting

Created by [Yongming Rao](https://raoyongming.github.io/)\*, [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Guangyi Chen](https://chengy12.github.io/), [Yansong Tang](https://andytang15.github.io/), [Zheng Zhu](http://www.zhengzhu.net/), Guan Huang, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), and [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1).

This repository contains PyTorch implementation for DenseCLIP (CVPR 2022).

DenseCLIP is a new framework for dense prediction by implicitly and explicitly leveraging the pre-trained knowledge from
CLIP. Specifically, we convert the original image-text matching
problem in CLIP to a pixel-text matching problem and
use the pixel-text score maps to guide the learning of dense
prediction models. By further using the contextual information
from the image to prompt the language model, we are
able to facilitate our model to better exploit the pre-trained
knowledge. Our method is model-agnostic, which can be
applied to arbitrary dense prediction systems and various
pre-trained visual backbones including both CLIP models
and ImageNet pre-trained models.

![intro](framework.gif)

Our code is based on mmsegmentation and mmdetection.

[[Project Page]](https://denseclip.ivg-research.xyz/) [[arXiv]](https://arxiv.org/abs/2112.01518)

## Usage

### Requirements

- torch>=1.8.0
- torchvision
- timm
- mmcv-full==1.3.17
- mmseg==0.19.0
- mmdet==2.17.0
- regex
- ftfy
- fvcore

To use our code, please first install the `mmcv-full` and `mmseg`/`mmdet` following the official guidelines ([`mmseg`](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md), [`mmdet`](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)) and prepare the datasets accordingly.

### Pre-trained CLIP Models

Download the pre-trained CLIP models (`RN50.pt`, `RN101.pt`, `VIT-B-16.pt`) and save them to the `pretrained` folder. The download links can be found in [the official CLIP repo](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30).

### Segmentation

#### Model Zoo
We provide DenseCLIP models for Semantic FPN framework.

| Model | FLOPs (G) | Params (M) | mIoU(SS) | mIoU(MS) | config | url |
|-------|-----------|------------|--------|--------|--------|-----| 
|RN50-CLIP|248.8|31.0|39.6|41.6|[config](segmentation/configs/fpn_clipres50_512x512_80k.py)|-| 
|RN50-DenseCLIP|269.2|50.3|43.5|44.7|[config](segmentation/configs/denseclip_fpn_res50_512x512_80k.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8636d4a95c60418ba63c/?dl=1)| 
|RN101-CLIP|326.6|50.0|42.7|44.3|[config](segmentation/configs/fpn_clipres101_512x512_80k.py)|-| 
|RN101-DenseCLIP|346.3|67.8|45.1|46.5|[config](segmentation/configs/denseclip_fpn_res101_512x512_80k.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/87387f3625ac42a68da5/?dl=1)| 
|ViT-B-CLIP|1037.4|100.8|49.4|50.3|[config](segmentation/configs/fpn_clipvit-b_640x640_80k.py)|-| 
|ViT-B-DenseCLIP|1043.1|105.3|50.6|51.3|[config](segmentation/configs/denseclip_fpn_vit-b_640x640_80k.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/d056a1a943214d479cd1/?dl=1)| 

#### Training & Evaluation on ADE20K

To train the DenseCLIP model based on CLIP ResNet-50, run:

```
bash dist_train.sh configs/denseclip_fpn_res50_512x512_80k.py 8
```

To evaluate the performance with multi-scale testing, run:

```
bash dist_test.sh configs/denseclip_fpn_res50_512x512_80k.py /path/to/checkpoint 8 --eval mIoU --aug-test
```

To better measure the complexity of the models, we provide a tool based on `fvcore` to accurately compute the FLOPs of `torch.einsum` and other operations:
```
python get_flops.py /path/to/config --fvcore
```
You can also remove the `--fvcore` flag to obtain the FLOPs measured by `mmcv` for comparisons.

###  Detection

#### Model Zoo
We provide models for both RetinaNet and Mask-RCNN framework.

##### RetinaNet
| Model | FLOPs (G) | Params (M) | box AP | config | url |
|-------|-----------|------------|--------|--------|-----| 
|RN50-CLIP|265|38|36.9|[config](detection/configs/retinanet_clip_r50_fpn_1x_coco.py)|-| 
|RN50-DenseCLIP|285|60|37.8|[config](detection/configs/retinanet_denseclip_r50_fpn_1x_coco.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/bfb64768d2124e99b79c/?dl=1)| 
|RN101-CLIP|341|57|40.5|[config](detection/configs/retinanet_clip_r101_fpn_1x_coco.py)|-| 
|RN101-DenseCLIP|360|78|41.1|[config](detection/configs/retinanet_denseclip_r101_fpn_1x_coco.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/cfb8cdf85dfb453eb786/?dl=1)| 

##### Mask R-CNN
| Model | FLOPs (G) | Params (M) | box AP | mask AP | config | url |
|-------|-----------|------------|--------|---------|--------|-----| 
|RN50-CLIP|301|44|39.3|36.8|[config](detection/configs/mask_rcnn_clip_r50_fpn_1x_coco.py)|-| 
|RN50-DenseCLIP|327|67|40.2|37.6|[config](detection/configs/mask_rcnn_denseclip_r50_fpn_1x_coco.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4adf197e693e4480bf26/?dl=1)| 
|RN101-CLIP|377|63|42.2|38.9|[config](detection/configs/mask_rcnn_clip_r101_fpn_1x_coco.py)|-| 
|RN101-DenseCLIP|399|84|42.6|39.6|[config](detection/configs/mask_rcnn_denseclip_r101_fpn_1x_coco.py)|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/ca072b19676942c3be82/?dl=1)| 



#### Training & Evaluation on COCO
To train our DenseCLIP-RN50 using RetinaNet framework, run
```bash
 bash dist_train.sh configs/retinanet_denseclip_r50_fpn_1x_coco.py 8
```

To evaluate the box AP of RN50-DenseCLIP (RetinaNet), run
```bash
bash dist_test.sh configs/retinanet_denseclip_r50_fpn_1x_coco.py /path/to/checkpoint 8 --eval bbox
```
To evaluate both the box AP and the mask AP of RN50-DenseCLIP (Mask-RCNN), run
```bash
bash dist_test.sh configs/mask_rcnn_denseclip_r50_fpn_1x_coco.py /path/to/checkpoint 8 --eval bbox segm
```

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{rao2021denseclip,
  title={DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting},
  author={Rao, Yongming and Zhao, Wenliang and Chen, Guangyi and Tang, Yansong and Zhu, Zheng and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
