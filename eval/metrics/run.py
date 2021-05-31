"""run.

Usage:
  run.py --true_path=<n> --pred_path=<n> --save_path=<n> --iou_thresh=<n>
  run.py (-h | --help)
  run.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --true_path=<n>    Root path to where the ground-truth is saved.
  --pred_path=<n>    Root path to where the predictions are saved.
  --save_path=<n>    Path where the prediction CSV files will be saved.
  --iou_thresh=<n>   Threshold on IOU
"""


import docopt
import os
import numpy as np
import pandas as pd
from utils import get_fast_pq, remap_label, binarize
from tabulate import tabulate
from tqdm import trange
tissue_types = [
                'Adrenal_gland',
                'Bile-duct',
                'Bladder',
                'Breast',
                'Cervix',
                'Colon',
                'Esophagus',
                'HeadNeck',
                'Kidney',
                'Liver',
                'Lung',
                'Ovarian',
                'Pancreatic',
                'Prostate',
                'Skin',
                'Stomach',
                'Testis',
                'Thyroid',
                'Uterus'
                ]

def main(args):
    """
    This function returns the statistics reported on the PanNuke dataset, reported in the paper below:

    Gamper, Jevgenij, Navid Alemi Koohbanani, Simon Graham, Mostafa Jahanifar, Syed Ali Khurram,
    Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot.
    "PanNuke Dataset Extension, Insights and Baselines." arXiv preprint arXiv:2003.10778 (2020).

    Args:
    Root path to the ground-truth
    Root path to the predictions
    Path where results will be saved

    Output:
    Terminal output of bPQ and mPQ results for each class and across tissues
    Saved CSV files for bPQ and mPQ results for each class and across tissues
    """

    true_root = args['--true_path']
    pred_root = args['--pred_path']
    save_path = args['--save_path']
    # I added this argument for conveinece
    iou_thresh = args['--iou_thresh'] # threshold on IOU

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    true_path = os.path.join(true_root,'masks.npy')  # path to the GT for a specific split
    pred_path = os.path.join(pred_root, 'masks.npy')  # path to the predictions for a specific split
    types_path = os.path.join(true_root,'types.npy') # path to the nuclei types
    iou_thresh = float(iou_thresh)
    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)
    types = np.load(types_path)

    mPQ_all = []
    bPQ_all = []
    
    # instead of simple range i added trange to see progress
    # loop over the images
    for i in trange(true.shape[0], desc='Evaluating'):
        pq = []
        pred_bin = binarize(pred[i,:,:,:5])
        true_bin = binarize(true[i,:,:,:5])

        if len(np.unique(true_bin)) == 1:
            pq_bin = np.nan # if ground truth is empty for that class, skip from calculation
        else:
            [_, _, pq_bin], _ = get_fast_pq(true_bin, pred_bin, match_iou=iou_thresh) # compute PQ

        # loop over the classes
        for j in range(5):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('int32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('int32')
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                pq_tmp = np.nan # if ground truth is empty for that class, skip from calculation
            else:
                [_, _, pq_tmp] , _ = get_fast_pq(true_tmp, pred_tmp, match_iou=iou_thresh) # compute PQ

            pq.append(pq_tmp)

        mPQ_all.append(pq)
        bPQ_all.append([pq_bin])

    # using np.nanmean skips values with nan from the mean calculation
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    neo_PQ = np.nanmean([pq[0] for pq in mPQ_all])
    inflam_PQ = np.nanmean([pq[1] for pq in mPQ_all])
    conn_PQ = np.nanmean([pq[2] for pq in mPQ_all])
    dead_PQ = np.nanmean([pq[3] for pq in mPQ_all])
    nonneo_PQ = np.nanmean([pq[4] for pq in mPQ_all])
    
    print('%'*40)
    print('Printing calculated metrics on a single split')
    print('%'*40)
    nuclei_type = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Non-Neoplastic']
    nuclei_type_pq = np.array([neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ])
    
    nuclei_result = np.concatenate((np.asarray(nuclei_type).reshape(-1,1), \
                                    np.round(np.asarray(nuclei_type_pq).reshape(-1,1),4)), 1)
        
    print(tabulate(np.ndarray.tolist(nuclei_result), headers = ["Nuclei Type", "PQ"], tablefmt="github"))
    print('%'*40)
    # Save per-class metrics as a csv file
    for_dataframe = {'Class Name': ['Neoplastic', 'Inflam', 'Connective', 'Dead', 'Non-Neoplastic'],
                        'PQ': [neo_PQ, conn_PQ, conn_PQ, dead_PQ, nonneo_PQ]}
    df = pd.DataFrame(for_dataframe, columns=['Tissue name', 'PQ'])
    df.to_csv(save_path + '/class_stats.csv')

    # Print for each tissue
    all_tissue_mPQ = []
    all_tissue_bPQ = []
    for tissue_name in tissue_types:
        indices = [i for i, x in enumerate(types) if x == tissue_name]
        tissue_PQ = [mPQ_each_image[i] for i in indices]
        tissue_PQ_bin = [bPQ_each_image[i] for i in indices]
        all_tissue_mPQ.append(np.nanmean(tissue_PQ))
        all_tissue_bPQ.append(np.nanmean(tissue_PQ_bin))

    # Save per-tissue metrics as a csv file
    for_dataframe = {'Tissue name': tissue_types + ['mean'],
                        'PQ': all_tissue_mPQ + [np.nanmean(all_tissue_mPQ)] , 'PQ bin': all_tissue_bPQ + [np.nanmean(all_tissue_bPQ)]}
    df = pd.DataFrame(for_dataframe, columns=['Tissue name', 'PQ', 'PQ bin'])
    df.to_csv(save_path + '/tissue_stats.csv')

    bPQ = all_tissue_bPQ
    mPQ = all_tissue_mPQ
    bPQ.append(np.nanmean(all_tissue_bPQ))
    mPQ.append(np.nanmean(all_tissue_mPQ))
    tissue_types.append('Average')
    
    result = np.concatenate((np.asarray(tissue_types).reshape(-1,1), \
                             np.round(np.asarray(mPQ).reshape(-1,1), 4), \
                             np.round(np.asarray(bPQ).reshape(-1,1),4)), 1)
        
    print(tabulate(np.ndarray.tolist(result), headers = ["Tissue Type", "mPQ", "bPQ"], tablefmt="github"))

#####
if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='PanNuke Evaluation v1.0')
    main(args)
