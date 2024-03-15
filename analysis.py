from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import pandas as pd
import argparse
import numpy as np
from scipy.optimize import curve_fit
import random
import os
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.ops import box_convert
import torch
from PIL import Image
from tqdm import tqdm
import supervision as sv


def get_args():
    parser = argparse.ArgumentParser(description='Run analysis on csv file')
    parser.add_argument('--threshold', type=float, help='Threshold for align_gt_norm')
    parser.add_argument('--key', type=str, choices=['masked', 'cropped'], help='Key for data')
    parser.add_argument('--dataset', type=str, choices=['agiqa-3k', 'aigciqa2023'], help='Dataset to use')
    parser.add_argument('--BOX_THRESHOLD', type=float, help='Box threshold to use')
    return parser.parse_args()

def run_analysis(args):
    # Load complete_grounding_df
    complete_df_path = f'/home/ece/Abhishek-Iyer1/ai-generated-iqa/src/results/{args.dataset}/grounding_qformer/predictions/base_blipTxtFeat_masked_valQ_gAdd_qk_identity_dummyTanh_lmean_concatA/best_model_results_unfiltered_batch_size_16_epochs_5_fold_Fold 5_lr_0.0001_masked_data_correctedv2_cos_adamw_0.1_noFloat.csv'
    complete_grounding_df = pd.read_csv(complete_df_path)
    csv_path = f'{args.key}_data_{args.BOX_THRESHOLD}/{args.dataset}/data.csv'
    # Load csv file into df
    df = pd.read_csv(csv_path)
    print(complete_grounding_df.columns, complete_grounding_df.head(), len(complete_grounding_df), len(df))
    # Check for mi_loc column for where samples are empty
    df = complete_grounding_df[~complete_grounding_df['name'].isin(df['name'])]
    # Filter above samples for where align_gt_norm is greater than threshold
    print(f"Found {len(df)} samples with no masked images")
    df = df[df['mos_align'] > args.threshold]
    print(df.head())

    print(f"Found {len(df)} samples with align_gt_norm > {args.threshold}")
if __name__ == '__main__':
    args = get_args()
    run_analysis(args)