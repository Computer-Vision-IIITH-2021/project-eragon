from PIL import Image
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
from SensorData import SensorData
import pickle
from multiprocessing import Pool
from zipfile import ZipFile
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

map_to_eigen13 = pickle.load(open('map_to_eigen13.pkl', 'rb'))

root_data_dir = '/mnt/storage2/data/datasets.rrc.iiit.ac.in/ScanNet/v2/scans'
existing_folders = os.listdir(root_data_dir)
tgt_folders = ['scene00{:02d}_{:02d}'.format(i, j) for i in range(60) for j in range(4)]
tgt_folders = [foldername for foldername in tgt_folders if foldername in existing_folders]

FRAMESKIP = 5
POOL_NUM_WORKERS = 8

os.makedirs('data/imgs', exist_ok=True)
os.makedirs('data/depths', exist_ok=True)
os.makedirs('data/masks', exist_ok=True)
os.makedirs('data/poses', exist_ok=True)
os.makedirs('data/intrinsics', exist_ok=True)


def extract_img_data(sd, foldername, idx):
    color_data = sd.frames[idx].decompress_color(sd.color_compression_type)
    io.imsave(os.path.join('data/imgs', foldername, '{}.png'.format(idx // FRAMESKIP)), color_data,
              check_contrast=False)

    depth_data = sd.frames[idx].decompress_depth(sd.depth_compression_type)
    depth_data = np.fromstring(depth_data, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
    np.save(os.path.join('data/depths', foldername, '{}.npy'.format(idx // FRAMESKIP)), depth_data)


def extract_mask_data(arg):
    zippath, foldername, filepath = arg.split(' ')
    zf = ZipFile(zippath)
    filename = os.path.basename(filepath)
    fileidx = int(os.path.splitext(filename)[0])
    with zf.open(filepath, 'r') as fp:
        img = io.imread(fp.read(), plugin='imageio')
        mapped_img = np.vectorize(map_to_eigen13.__getitem__)(img).astype(np.uint8)
        io.imsave(os.path.join('data/masks', foldername, '{}.png'.format(fileidx // FRAMESKIP)),
                  mapped_img, check_contrast=False)


def preprocess_data(foldername):
    print(foldername)
    os.makedirs(os.path.join('data/imgs', foldername), exist_ok=True)
    os.makedirs(os.path.join('data/depths', foldername), exist_ok=True)
    os.makedirs(os.path.join('data/masks', foldername), exist_ok=True)
    os.makedirs(os.path.join('data/poses', foldername), exist_ok=True)
    os.makedirs(os.path.join('data/intrinsics', foldername), exist_ok=True)

    st = time.time()
    sd = SensorData(os.path.join(root_data_dir, foldername, '{}.sens'.format(foldername)))
    sd_loading_time = time.time() - st
    sd.export_poses(os.path.join('data/poses', foldername), FRAMESKIP)
    sd.export_intrinsics(os.path.join('data/intrinsics', foldername))

    st = time.time()
    frame_indices = list(range(0, len(sd.frames), FRAMESKIP))
    sd.extract_color_depth_data(frame_indices, FRAMESKIP)
    color_depth_extract_time = time.time() - st

    zip_path = os.path.join(root_data_dir, foldername, '{}_2d-label-filt.zip'.format(foldername))
    st = time.time()
    filepaths = []
    with ZipFile(zip_path) as zf:
        for filepath in zf.namelist():
            if os.path.splitext(filepath)[1] != '.png':
                continue
            filename = os.path.basename(filepath)
            fileidx = int(os.path.splitext(filename)[0])
            if fileidx % FRAMESKIP != 0:
                continue
            filepaths.append('{} {} {}'.format(zip_path, foldername, filepath))
    with ProcessPoolExecutor(max_workers=POOL_NUM_WORKERS) as processpool:
        processpool.map(extract_mask_data, filepaths)
    mask_extract_time = time.time() - st

    print(foldername, sd_loading_time, color_depth_extract_time, mask_extract_time)


for foldername in tgt_folders:
    preprocess_data(foldername)
