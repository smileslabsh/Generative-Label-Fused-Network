## Generative-Label-Fused-Network
Generative Label Fused Network for Image-Text Matching

## Introduction

This is the source code of Generative Label Fused Network, an approch for Image-Text Matching. It is built on top of the SCAN (by [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN)) and PFAN (by  [Yaxiong Wang]( [HaoYang0123/Position-Focused-Attention-Network: Position Focused Attention Network for Image-Text Matching (github.com)](https://github.com/HaoYang0123/Position-Focused-Attention-Network) ))in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 0.3
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

The workflow of GLFN

<img src="https://raw.githubusercontent.com/smileslabsh/Generative-Label-Fused-Network/main/figures/main.png" width="745" alt="workflow" /> 

## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The position information of images can be downloaded from [here](https://github.com/HaoYang0123/Position-Focused-Attention-Network/tree/master) 

## Training new models

To train Flickr30K and MS-COCO models:
```bash
sh run_train.sh
```
## Results
|            | i2t-1    |i2t-5    |i2t-10    |t2i-1    |t2i-5    |t2i-10    |
| :---------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Flickr30K | 75.1  | 93.8  |  97.2  | 54.5  | 82.8  |  89.9  |
| MSCOCO | 78.4  | 96.0  |  98.5  | 62.6  | 89.6  |  95.4  |
