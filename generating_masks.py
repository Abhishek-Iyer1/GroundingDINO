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

    parser = argparse.ArgumentParser('Title')

    parser.add_argument('--dataset', required=True, type=str, choices=['agiqa-3k', 'aigciqa2023'], help='Parameter description')
    parser.add_argument('--filtered', required=True, type=str, choices=['filtered', 'unfiltered'])
    parser.add_argument('--seed', default=1, type=int, help='Parameter description')
    parser.add_argument('--num_images', default=10, type=int, help='Parameter description')
    parser.add_argument('--feature', default='mask', choices=['mask', 'crop'], type=str, help='Parameter description')
    parser.add_argument('--pipeline', required=True, choices=['preprocess', 'annotate'], type=str, help='Parameter description')
    parser.add_argument('--with_border', action='store_true', help='Parameter description')

    return parser.parse_args()

class NonLinearFit():
    def __init__(self, x=None, y=None):
        self.maxfev = 10000
        self.nlinfunc = self.four_param_logistic_QA
        if x.any():
            beta_init = [np.min(x), np.max(x), np.mean(y), 1]
            self.params, self.params_covariance = curve_fit(self.nlinfunc, y, x, p0=beta_init, maxfev=self.maxfev)
    
    def fit(self, x, y):
        beta_init = [np.min(x), np.max(x), np.mean(y), 1]
        self.params, self.params_covariance = curve_fit(self.nlinfunc, y, x, p0=beta_init, maxfev=self.maxfev)
    
    def transform(self, y):
        fitted_predictions = self.nlinfunc(y, *self.params)
        return fitted_predictions
    
    def four_param_logistic_QA(self, x, beta1, beta2, beta3, beta4):
        return ((beta1 - beta2)/(1 + (np.exp((beta3-x)/np.abs(beta4))))) + beta2
    
    def exponential(self, x, beta1, beta2, beta3):
        return beta1 + beta2 * np.exp(beta3 * x)
    
    def affine(self, x, beta1, beta2):
        return beta1 + beta2 * x

def add_line_breaks(text, char_count):
    lines = []
    current_line = []

    for word in text.split():
        if current_line and len(' '.join(current_line + [word])) > char_count:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    result = '\n'.join(lines)

    return result


def generate_masks(args):

    # random.seed(args.seed)

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    HIGH_THRESH = 0.75
    LOW_THRESH = 0.5
    num_images = args.num_images

    if args.dataset == 'agiqa-3k':
        df_path = '/home/ece/nithinc_datasets/AGIQA-3k/data.csv'
        img_prefix = '/home/ece/nithinc_datasets/AGIQA-3k/Images/'
        df = pd.read_csv(df_path)
        if args.filtered == 'filtered':
            discard_models_condition = ((df["name"].str.startswith("AttnGAN")) | (df["name"].str.startswith("glide")))
            df = df[(np.bitwise_not(discard_models_condition))]

    elif args.dataset == 'aigciqa2023':
        df_path = '/home/ece/nithinc_datasets/AIGCIQA2023/aigciqa_all_data.csv'
        img_prefix = '/home/ece/nithinc_datasets/AIGCIQA2023/allimg'
        df = pd.read_csv(df_path)
        df = df.rename(columns={'im_loc': 'name', 'mosz1': 'mos_quality', 'mosz3': 'mos_align'})
        if args.filtered == 'filtered':
            discard_models_condition = ((df["name"].str.startswith("Lafite")) | (df["name"].str.startswith("Glide")))
            df = df[(np.bitwise_not(discard_models_condition))]
    
    key = 'cropped' if args.feature == 'crop' else 'masked'
    
    # Construct a Sample
    # index = random.sample(range(0, len(df)), 1)
    final_df = pd.DataFrame({
        'name': [],
        'prompt': [],
        'mos_align': [],
        'boxes': [],
        'logits': [],
        'phrases': [],
        f'{key}_image_loc': []
    })

    for _, row in tqdm(df.iterrows()):
        
        temp_dict = {
            'name': [],
            'prompt': [],
            'mos_align': [],
            'boxes': [],
            'logits': [],
            'phrases': [],
            f'{key}_image_loc': []
        }

        image_source, image = load_image(os.path.join(img_prefix, row['name']), args)
        prompt = row.loc['prompt']
        label = row.loc['mos_align']

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        if args.dataset == 'agiqa-3k':
            row_name = "_".join(row["name"].split(".")[:-1])
        elif args.dataset == 'aigciqa2023':
            row_name = "_".join([row['model'], *row["name"].split(".")[:-1]])

        os.makedirs(f'{key}_data_{BOX_THRESHOLD}/{args.dataset}/{key}_images/{row_name}', exist_ok=True)
        # print(f"No. of Boxes: {len(boxes)}")
        if args.feature == 'mask':
            features = mask(boxes, image_source, args) # (len(boxes), image_source.shape[0], image_source.shape[1])
            # print(f"No. of Masked Features: {len(features)}")
        else:
            features = crop(boxes, image_source, args)
            # print(f"No. of Cropped Features: {len(features)}")

        assert len(boxes) == len(features) == len(phrases) == len(logits), f"Len Boxes: {len(boxes)} is not equal to Len Features: {len(features)}"

        for idx, b, m_i, l, p in list(zip(range(len(boxes)), boxes, features, logits, phrases)):
            temp_dict['name'].append(row['name'])
            temp_dict['prompt'].append(prompt)
            temp_dict['mos_align'].append(label)
            temp_dict['boxes'].append(b.data)
            temp_dict['phrases'].append(p)
            temp_dict['logits'].append(l.data)
            mi_name = f'{idx+1}.jpg'
            mi_path = f'{key}_data_{BOX_THRESHOLD}/{args.dataset}/{key}_images/{row_name}/{mi_name}'
            temp_dict[f'{key}_image_loc'].append(mi_path)
            m_i_img = Image.fromarray(m_i)
            m_i_img.save(mi_path)

        temp_df = pd.DataFrame(temp_dict)
        
        final_df = pd.concat((final_df, temp_df), axis = 0)
        # print(f"Temp df: {temp_df}, Len: {len(temp_df)}, {len(boxes)}, {len(features)}, {len(phrases)}, {len(logits)}")
        # if row["name"] == "sd1.5_highcorr_096.jpg":
        #     input()

    final_df.to_csv(f'{key}_data_{BOX_THRESHOLD}/{args.dataset}/data.csv')




        
    # Store details in a CSV File (Image Name, Phrases, logits, Bbox, Path to Masked Image)

    # Predict every sample and store results
            
def annotate_dataset(args):
    
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    if args.dataset == 'agiqa-3k':
        df_path = '/home/ece/nithinc_datasets/AGIQA-3k/data.csv'
        img_prefix = '/home/ece/nithinc_datasets/AGIQA-3k/Images'
        df = pd.read_csv(df_path)
        if args.filtered == 'filtered':
            discard_models_condition = ((df["name"].str.startswith("AttnGAN")) | (df["name"].str.startswith("glide")))
            df = df[(np.bitwise_not(discard_models_condition))]

    elif args.dataset == 'aigciqa2023':
        df_path = '/home/ece/nithinc_datasets/AIGCIQA2023/aigciqa_all_data.csv'
        img_prefix = '/home/ece/nithinc_datasets/AIGCIQA2023/allimg'
        df = pd.read_csv(df_path)
        df = df.rename(columns={'im_loc': 'name', 'mosz1': 'mos_quality', 'mosz3': 'mos_align'})
        if args.filtered == 'filtered':
            discard_models_condition = ((df["name"].str.startswith("Lafite")) | (df["name"].str.startswith("Glide")))
    save_path = f"annotated_data/{args.dataset}/annotated_images"
    # save_path = os.path.join(*img_prefix.split("/")[:-1], "annotated_images")
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    for _, row in tqdm(df.iterrows()):
        image_source, image = load_image(os.path.join(img_prefix, row['name']), args)
        prompt = row.loc['prompt']
        label = row.loc['mos_align']

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        annotated_image = annotate(image_source, boxes, logits, phrases)
        # Save image
        annotated_image_pil = Image.fromarray(annotated_image[:, :, ::-1])
        annotated_image_pil.save(os.path.join(save_path, row['name']))

def mask(boxes, image_source, args):
    clip_upper = 612 if args.with_border else 512
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    rounded_xyxy = np.zeros_like(xyxy, dtype=int)

    for i in range(len(detections)):
        rounded_xyxy[i] = np.array([int(elem) for elem in detections.xyxy[i]])
    rounded_xyxy = np.clip(rounded_xyxy, 0, clip_upper)
    masks = np.zeros((len(boxes), *image_source.shape[:2]), np.uint8)
    masked_images = []

    for m, b in list(zip(masks, rounded_xyxy)):
        m[b[1]:b[3], b[0]:b[2]] = 1
        m_i = cv2.bitwise_and(image_source,image_source, mask=m)
        masked_images.append(m_i)

    return masked_images

def crop(boxes, image_source, args):
    clip_upper = 612 if args.with_border else 512
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    rounded_xyxy = np.zeros_like(xyxy, dtype=int)
    for i in range(len(detections)):
        rounded_xyxy[i] = np.array([int(elem) for elem in detections.xyxy[i]])
    rounded_xyxy = np.clip(rounded_xyxy, 0, clip_upper)
    crops = []
    print(f'Rounded xyxy: {rounded_xyxy}')
    for b in rounded_xyxy:
        crops.append(image_source[b[1]:b[3], b[0]:b[2]])

    return crops

def normalize(dataframes, args):
    train_df, val_df, test_df = dataframes
    mos_quality_key = 'mos_quality' #if args.dataset == 'agiqa-3k' else 'mosz1'
    mos_alignment_key = 'mos_align' #if args.dataset == 'agiqa-3k' else 'mosz3'

    mos_q_scores = train_df[mos_quality_key].tolist() + val_df[mos_quality_key].tolist() + test_df[mos_quality_key].tolist()
    mos_a_scores = train_df[mos_alignment_key].tolist() + val_df[mos_alignment_key].tolist() + test_df[mos_alignment_key].tolist()

    mos_a_min = min(mos_a_scores)
    mos_a_max = max(mos_a_scores)
    mos_q_min = min(mos_q_scores)
    mos_q_max = max(mos_q_scores)

    # normalize the scores
    train_df['quality_gt_norm'] = (train_df[mos_quality_key] - mos_q_min) / (mos_q_max - mos_q_min)
    val_df['quality_gt_norm'] = (val_df[mos_quality_key] - mos_q_min) / (mos_q_max - mos_q_min)
    test_df['quality_gt_norm'] = (test_df[mos_quality_key] - mos_q_min) / (mos_q_max - mos_q_min)

    train_df['align_gt_norm'] = (train_df[mos_alignment_key] - mos_a_min) / (mos_a_max - mos_a_min)
    val_df['align_gt_norm'] = (val_df[mos_alignment_key] - mos_a_min) / (mos_a_max - mos_a_min)
    test_df['align_gt_norm'] = (test_df[mos_alignment_key] - mos_a_min) / (mos_a_max - mos_a_min)

    return train_df, val_df, test_df
    
if __name__ == '__main__':
    args = get_args()
    if args.pipeline == 'preprocess':
        generate_masks(args)
    elif args.pipeline == 'annotate':
        annotate_dataset(args)