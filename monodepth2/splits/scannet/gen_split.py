import os
import numpy as np

np.random.seed(0)

data_dir = '/mnt/storage/Projects/Pytorch-UNet/data/imgs'

scene_folders = list(set([i.split('_')[0] for i in os.listdir(data_dir)]))
np.random.shuffle(scene_folders)
train_size = np.floor(len(scene_folders) * 0.7).astype(int)
train_folders = scene_folders[:train_size]
val_folders = scene_folders[train_size:]

with open('train_files.txt', 'w') as train_txt:
    for folder in os.listdir(data_dir):
        if np.any([i in folder for i in train_folders]):
            for f in os.listdir(os.path.join(data_dir, folder)):
                print(f)
                fileidx = int(os.path.splitext(f)[0])
                if fileidx == 0:
                    continue
                if fileidx == len(os.listdir(os.path.join(data_dir, folder))) - 1:
                    continue
                train_txt.write('{} {}\n'.format(folder, fileidx))

with open('val_files.txt', 'w') as val_txt:
    for folder in os.listdir(data_dir):
        if np.any([i in folder for i in val_folders]):
            for f in os.listdir(os.path.join(data_dir, folder)):
                print(f)
                fileidx = int(os.path.splitext(f)[0])
                if fileidx == 0:
                    continue
                if fileidx == len(os.listdir(os.path.join(data_dir, folder))) - 1:
                    continue
                val_txt.write('{} {}\n'.format(folder, fileidx))
