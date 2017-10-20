import numpy as np
import argparse
from  dataloader.image_folder import make_dataset
import os
from PIL import Image
from collections import Counter
import pickle

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--split', type=str, default='eigen', help='data split')
parser.add_argument('--predicted_depth_path', type=str, default='../dataset/predicted', help='path to estimated depth')
parser.add_argument('--gt_path', type=str, default='../dataset/groundtruth', help='path to ground truth depth')
parser.add_argument('--min_depth', type=float, default=1e-3, help='minimun depth for evaluation')
parser.add_argument('--max_depth', type=float, default=8.0, help='maximun depth for evaluation, indoor 8.0, outdoor 80')
parser.add_argument('--eigen_crop', action='store_true', help='if set, crops according to Eigen NIPS14')

args = parser.parse_args()

def compute_errors(ground_truth, predication):

    # accuracy
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.sqrt(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_indoor_depth(file_path):
    depths = []
    dataset = make_dataset(file_path)
    for data in dataset:
        depth = Image.open(data)
        depth = depth.resize((425,425),Image.BICUBIC)
        depth = np.array(depth)
        depth = depth.astype(np.float32) / 255 * 8.0
        depths.append(depth)
    return depths

if __name__ == '__main__':

    ground_truth = load_indoor_depth(args.gt_path)
    predicted_truth = load_indoor_depth(args.predicted_depth_path)

    num_samples = len(ground_truth)

    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples,np.float32)
    rmse = np.zeros(num_samples,np.float32)
    rmse_log = np.zeros(num_samples,np.float32)
    a1 = np.zeros(num_samples,np.float32)
    a2 = np.zeros(num_samples,np.float32)
    a3 = np.zeros(num_samples,np.float32)

    for i in range(len(ground_truth)):
        ground = ground_truth[i]
        predicted = predicted_truth[i]
        mask = np.logical_and(ground > args.min_depth, ground < args.max_depth)
        abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(ground[mask],predicted[mask])

    print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
    print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
           .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))