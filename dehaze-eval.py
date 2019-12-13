import argparse
import glob
import os
import re

import cv2
import numpy as np
from skimage import io
from skimage.color import deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from skimage.util import img_as_float32

gt_suffix = '_GT.png'
in_suffix = '_hazy.png_final.jpg'

parser = argparse.ArgumentParser(description='Evaluate dehazed images against their ground truth.')
parser.add_argument('groundtruth', help='directory with ground truth images with suffix ' + gt_suffix)
parser.add_argument('input', help='directory with images to compare to, same filename as originals but with suffix ' + in_suffix)
args = parser.parse_args()

groundtruths = glob.glob(args.groundtruth + '*' + gt_suffix)
inputs = glob.glob(args.input + '*' + in_suffix)

assert len(groundtruths) == len(inputs), "Number of matched images in directories doesn't match."

def get_key(path: str, suffix: str) -> int:
    return int(re.search(r'(\d{2})' + suffix, path).group(1))

groundtruths = { get_key(path, gt_suffix) : path for path in groundtruths }
inputs = { get_key(path, in_suffix) : path for path in inputs }

psnr_sum = 0
ssim_sum = 0
deltaE_sum = 0

for key in inputs:
    gt = img_as_float32(io.imread(groundtruths[key]))
    dehazed = img_as_float32(io.imread(inputs[key]))
    gt_resized = resize(gt, dehazed.shape, preserve_range=True, anti_aliasing=False)
    psnr = peak_signal_noise_ratio(gt_resized, dehazed)
    ssim = structural_similarity(gt_resized, dehazed, multichannel=True)
    gt_lab = cv2.cvtColor(gt_resized, cv2.COLOR_RGB2Lab)
    in_lab = cv2.cvtColor(dehazed, cv2.COLOR_RGB2Lab)
    deltaE = np.average(deltaE_ciede2000(gt_lab, in_lab))
    print("{:02} - PSNR: {:.6f} SSIM: {:.6f} CIEDE2000: {:.6f}".format(key, psnr, ssim, deltaE))
    psnr_sum += psnr
    ssim_sum += ssim
    deltaE_sum += deltaE

print("AVG over {} images - PSNR: {:.6f} SSIM: {:.6f} CIEDE2000: {:.6f}"
    .format(len(inputs), psnr_sum/len(inputs), ssim_sum/len(inputs), deltaE_sum/len(inputs)))
