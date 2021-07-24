<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FTSFD-Net-for-Nuclei-Segmentation-and-Classification&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# TSFD-Net-for-Nuclei-Segmentation-and-Classification

A growing interest is emerging for cancer diagnosis from digital histopathology images. These images typically contain millions of nuclei from a single tissue slide. The appearance, shape, texture, and morphological features of the nuclei depend upon the tissue types. Segmentation, detection, and classification of the nuclei are the core analyzing steps for digital microscopic images. Therefore, in this paper, we design a Tissue Specific Feature Distillation Network (TSFD-Net) that extracts
distilled features from specific tissue types, generates semantically strong feature pyramids by fus-ng the multi-scale features of the backbone, and predicts the semantic segmentation and boundarydetection masks. The average multi-class panoptic quality (mPQ) of tissue types and nuclei categories of the TSFD-Net for semantic and instance segmentation tasks on the PanNuke dataset are 4.15%, 4.11%, and 3.4% better than the state-of-the-art StarDIST, Hover-Net, and CPP-Net, respec-
tively. Moreover, TSFD-Net outperforms the Mask-RCNN, Micro-Net, and Hover-Net in precision, recall, and F1-score matrices for detection and classification of nuclei types. The inference time of the TSFD-Net is 13.76, 2.06, and 2.23 times less than that of the Hover-Net, StarDIST, and CPP-Net, respectively

## Pannuke Dataset:

The PanNuke dataset can be downloaded form [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke).


## Network Architecture:

Figure below shows full architecture of our proposed TSFD-Net.

![alt text](https://github.com/Mr-TalhaIlyas/TSFD-Net-for-Nuclei-Segmentation-and-Classification/blob/master/screens/img1.png)


## Results

The Table below compare quantitative results of different models.

![alt text](https://github.com/Mr-TalhaIlyas/TSFD-Net-for-Nuclei-Segmentation-and-Classification/blob/master/screens/img2.png)

## Visual Results
The figure below shows some qualitative results.

![alt text](https://github.com/Mr-TalhaIlyas/TSFD-Net-for-Nuclei-Segmentation-and-Classification/blob/master/screens/img3.png)

## Evaluation

To evaluate the model we used the Panoptic Quality metric as introduced in [HoverNet](https://www.sciencedirect.com/science/article/pii/S1361841519301045) paper.

We use the official implementation provided by the authors of [Pannuke](https://jgamper.github.io/PanNukeDataset/) dataset.

To see our implementation follow the [link](https://github.com/Mr-TalhaIlyas/TSFD-Net-for-Nuclei-Segmentation-and-Classification/tree/master/eval).
We mainly follow the original implementation with some minor improvements for exception handelling, bug fixes and better visualization.


## Citation 

```
Coming Soon IA...!

paper under review in Elsevier Neural Networks ;)
```



