import os
import numpy as np
import cv2
from skimage.segmentation import slic, felzenszwalb


from concurrent.futures import ProcessPoolExecutor
import tqdm
from functools import partial


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int,
                    help='number of cpu workers',
                    default=6)
parser.add_argument('--data_dir', type=str,
                    help='path to scannet data',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted segment',
                    required=True)
args = parser.parse_args()

output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def image2seg(folder, filename, algo='felzenszwalb'):
    image = cv2.imread(os.path.join(tgt_dir, folder, filename))
    image = cv2.resize(image, (640, 480))

    if algo == 'felzenszwalb':
        segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50).astype(np.int16)
    elif algo == 'slic':
        segment = slic(image, n_segments=500, sigma=0.5, compactness=1).astype(np.int16)
    else:
        raise NotImplementedError("Segmentation algorithm not implemented.")

    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    filename_wo_extn = os.path.splitext(os.path.basename(filename))[0]

    np.savez(os.path.join(output_dir, folder, "seg_{}.npz".format(filename_wo_extn)), segment_0=segment)
    return


executor = ProcessPoolExecutor(max_workers=args.num_workers)
futures = []

tgt_dir = os.path.join(args.data_dir, 'imgs')
all_files = [(folder, filename) for folder in os.listdir(tgt_dir) for filename in
             os.listdir(os.path.join(tgt_dir, folder))]

for folder, filename in all_files:
    task = partial(image2seg, folder, filename)
    futures.append(executor.submit(task))

results = []
[results.append(future.result()) for future in tqdm.tqdm(futures)]


