# PanNuke Evaluation Metrics

This repository calculates metrics on the PanNuke dataset, as reported [here](https://arxiv.org/abs/2003.10778): <br />

- **Binary PQ (bPQ)**: Assumes all nuclei belong to same class and reports the average PQ across tissue types. <br />
- **Multi-Class PQ (mPQ)**: Reports the average PQ across the classes and tissue types. <br />
- **Neoplastic PQ**: Reports the PQ for the neoplastic class on all tissues. <br />
- **Non-Neoplastic PQ**: Reports the PQ for the non-neoplastic class on all tissues. <br />
- **Inflammatory PQ**: Reports the PQ for the inflammatory class on all tissues. <br />
- **Connective PQ**: Reports the PQ for the connective class on all tissues. <br />
- **Dead PQ**: Reports the PQ for the dead class on all tissues. <br />


## Set up envrionment

```
conda create --name pannuke python==3.6
conda activate pannuke
pip install -r requirements.txt
```

## Running the Code 

Usage Example
````
!python /home/user01/data_ssd/Talha/pannuke/Eval/metrics/run.py \
    --true_path={gt_dir} \
    --pred_path={pred_dir} \
    --save_path={save_path} \
    --iou_thresh={0.5}
````

Options:
```
  -h --help          Show this string.
  --version          Show version.
  --true_path=<n>    Root path to where the ground-truth is saved.
  --pred_path=<n>    Root path to where the predictions are saved.
  --save_path=<n>    Path where the prediction CSV files will be saved.
  --iou_thresh=<n>   threshold on IOU to consider an instance to be TP.
```

Before running the code, ground truth and predictions must be saved in the following structure: <br />

- True Masks:
    - `<true_path>/masks.npy`
- True Tissue Types:
    - `<true_path>/types.npy`
- Prediction Masks:
    - `<pred_path>/masks.npy`

Here, prediction masks are saved in the same format as the true masks. i.e a single `Nx256x256xC` array, where `N` is the number of test images in that specific fold and `C` is the number of positive classes. The ordering of the channels from index `0` to `4` is `neoplastic`, `inflammatory`, `connective tissue`, `dead` and `non-neoplastic epithelial`.

## Sample Output

```
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Printing calculated metrics on a single split
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
| Nuclei Type    |     PQ |
|----------------|--------|
| Neoplastic     | 0.5724 |
| Inflammatory   | 0.4532 |
| Connective     | 0.4228 |
| Dead           | 0.2135 |
| Non-Neoplastic | 0.5663 |
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
| Tissue Type   |    mPQ |    bPQ |
|---------------|--------|--------|
| Adrenal_gland | 0.5223 | 0.69   |
| Bile-duct     | 0.5    | 0.6284 |
| Bladder       | 0.5738 | 0.6773 |
| Breast        | 0.5106 | 0.6245 |
| Cervix        | 0.5204 | 0.6561 |
| Colon         | 0.4382 | 0.537  |
| Esophagus     | 0.5438 | 0.6306 |
| HeadNeck      | 0.4937 | 0.6277 |
| Kidney        | 0.5517 | 0.6824 |
| Liver         | 0.5079 | 0.6675 |
| Lung          | 0.4274 | 0.5941 |
| Ovarian       | 0.5253 | 0.6431 |
| Pancreatic    | 0.4893 | 0.6241 |
| Prostate      | 0.5431 | 0.6406 |
| Skin          | 0.4354 | 0.6074 |
| Stomach       | 0.4871 | 0.6529 |
| Testis        | 0.4843 | 0.6435 |
| Thyroid       | 0.5154 | 0.6692 |
| Uterus        | 0.5068 | 0.6204 |
| Average       | 0.504  | 0.6377 |
```

## Referneces


```
@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benet, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
  organization={Springer}
}
```
```
@article{gamper2020pannuke,
  title={PanNuke Dataset Extension, Insights and Baselines},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Graham, Simon and Jahanifar, Mostafa and Khurram, Syed Ali and Azam, Ayesha and Hewitt, Katherine and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2003.10778},
  year={2020}
}
```





