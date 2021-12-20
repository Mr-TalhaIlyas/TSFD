<img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FTSFD&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# TSFD-Net-for-Nuclei-Segmentation-and-Classification

Nuclei segmentation and classification using hematoxylin and eosin-stained histology images is a challenging task due a variety of issues, such as color inconsistency resulting from non-uniform manual staining operations, clustering of nuclei and blurry and overlapping nuclei boundaries. Existing approaches involve segmenting nuclei by drawing their polygon representations or by measuring the distances between nuclei centroids. In contrast, we leverage the fact that morphological features (appearance, shape and texture) of nuclei vary greatly depending upon the tissue type on which it is located. We exploit this information by extracting tissue specific (TS) features from raw histopathology images using our tissue specific feature distillation (TSFD) backbone. Then our bi-directional feature pyramid network (BiFPN) generates a robust hierarchical feature pyramid using these TS features. Next, our interlinked decoders jointly optimize and fuse these features to generate final predictions. We also propose a novel loss combination for joint optimization and faster convergence of our proposed network. Extensive ablation studies are performed to validate the effectiveness of each component of TSFD-Net. TSFD- Net achieves state-of-the-art performance on PanNuke dataset having 19 different tissue types and up to 5 clinically important tumor classes. 

## Pannuke Dataset:

The PanNuke dataset can be downloaded form [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke).


## Network Architecture:

Figure below shows full architecture of our proposed TSFD-Net.

<center><img src = 'https://raw.githubusercontent.com/Mr-TalhaIlyas/TSFD/master/screens/img1.png?token=ARO52YNP4LKM7EZHMQ7JEITBX7RBY'></center>


## Results

The Table below compare quantitative results of different models.

![alt text](https://github.com/Mr-TalhaIlyas/TSFD/raw/master/screens/img2.png)

## Visual Results
The figure below shows some qualitative results.

![alt text](https://github.com/Mr-TalhaIlyas/TSFD/raw/master/screens/img3.png)

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



