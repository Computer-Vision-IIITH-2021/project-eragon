from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2
import random
import torch
from torchvision import transforms

from datasets.mono_dataset import MonoDataset

from datasets.keypt_extractors import get_keypts


class ScanNetDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(ScanNetDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1296, 968)
        self.num_pts = 3000

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1])

        inputs["filepath"] = self.filenames[index]

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, None, do_flip)

        inputs[("segment", 0, 0)] = self.to_tensor(self.get_segment(folder, frame_index, do_flip)).long() + 1

        img = np.array(inputs[("color", 0, -1)])
        keypts = get_keypts(img, "orb").astype(np.int32)
        if keypts.shape[0] == 0:
            keypts = get_keypts(img, "sift").astype(np.int32)  # slower, so used as fallback

        if keypts.shape[0] != 0:
            keypts[:, 0] = keypts[:, 0] * self.width // self.full_res_shape[0]
            keypts[:, 1] = keypts[:, 1] * self.height // self.full_res_shape[1]
        else:
            print("No keypoints detected for {}".format(line))
            keypts = np.array([[], []], dtype=np.int32).T

        remaining_pts = self.num_pts - keypts.shape[0]
        if remaining_pts > 0:
            random_pts = np.zeros((remaining_pts, 2), dtype=np.int32)
            random_pts[:, 0] = np.random.randint(0, self.width, remaining_pts)
            random_pts[:, 1] = np.random.randint(0, self.height, remaining_pts)
        else:
            random_pts = np.array([[], []], dtype=np.int32).T

        all_pts = np.concatenate([keypts, random_pts], axis=0)[:self.num_pts, :]

        pt_mask = np.zeros((self.height, self.width), dtype=np.int32)
        pt_mask[all_pts[:, 1], all_pts[:, 0]] = 1

        inputs["keypts"] = torch.from_numpy(all_pts).float()
        # inputs["keypt_mask"] = torch.from_numpy(pt_mask).float()

        # adjusting intrinsics to match each scale in the pyramid
        K = self.get_K(folder, np.array(inputs[("color", 0, -1)]).shape)
        for scale in range(self.num_scales):
            K = K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, None, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_K(self, folder, img_shape):
        camera_intrinsics_path = os.path.join(
            self.data_path,
            "intrinsics",
            folder,
            "intrinsic_color.txt")
        with open(camera_intrinsics_path, 'r') as f:
            intrinsics = f.readlines()

        intrinsics = np.array([i.strip('\n').split() for i in intrinsics]).astype(np.float32)
        intrinsics[0, :] /= img_shape[1]
        intrinsics[1, :] /= img_shape[0]
        return intrinsics

    def check_depth(self):
        if len(self.filenames) == 0:
            return False
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = int(line[1])

        depth_path = os.path.join(
            self.data_path,
            "depths",
            folder,
            "{}.npy".format(frame_index))

        return os.path.isfile(depth_path)

    def get_image_path(self, folder, frame_index):
        image_path = os.path.join(
            self.data_path,
            "imgs",
            folder,
            "{}.png".format(frame_index))
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_pose(self, folder, frame_index, do_flip):
        pose_path = os.path.join(
            self.data_path,
            "poses",
            folder,
            "{}.txt".format(frame_index))

        pose = np.loadtxt(pose_path).astype(np.float32)

        if do_flip:
            pass #todo implement flip for pose

        return pose

    def get_segment(self, folder, frame_index, do_flip):
        seg_path = os.path.join(
            self.data_path,
            "superpixels",
            folder,
            "seg_{}.npz".format(frame_index))

        segment = np.load(seg_path)['segment_0']

        segment = cv2.resize(segment, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if do_flip:
            segment = cv2.flip(segment, 1)

        return segment

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = os.path.join(
            self.data_path,
            "depths",
            folder,
            "{}.npy".format(frame_index))

        depth_gt = np.load(depth_path)
        depth_gt = (depth_gt * 255.0 / np.max(depth_gt)).astype(np.uint8)
        depth_gt = pil.fromarray(depth_gt, 'L').resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


if __name__ == '__main__':
    data_path = '/mnt/storage2/data/scannet'
    filenames = ['scene0048_01 76', 'scene0048_01 38', 'scene0048_01 31', 'scene0048_01 9', 'scene0048_01 53',
                 'scene0048_01 106', 'scene0048_01 88', 'scene0048_01 61', 'scene0048_01 82', 'scene0048_01 58',
                 'scene0048_01 28', 'scene0048_01 102', 'scene0048_01 66', 'scene0048_01 125', 'scene0048_01 60',
                 'scene0048_01 5', 'scene0048_01 93', 'scene0048_01 87', 'scene0048_01 114', 'scene0048_01 99',
                 'scene0048_01 80', 'scene0048_01 6', 'scene0048_01 105', 'scene0048_01 62', 'scene0048_01 57',
                 'scene0048_01 50', 'scene0048_01 27', 'scene0048_01 73', 'scene0048_01 126', 'scene0048_01 67',
                 'scene0048_01 54', 'scene0048_01 85', 'scene0048_01 13', 'scene0048_01 7', 'scene0048_01 113',
                 'scene0048_01 68', 'scene0048_01 17', 'scene0048_01 117', 'scene0048_01 1', 'scene0048_01 70',
                 'scene0048_01 26', 'scene0048_01 48', 'scene0048_01 15', 'scene0048_01 45', 'scene0048_01 132',
                 'scene0048_01 71', 'scene0048_01 130', 'scene0048_01 25', 'scene0048_01 29', 'scene0048_01 3',
                 'scene0048_01 19', 'scene0048_01 78', 'scene0048_01 97', 'scene0048_01 2', 'scene0048_01 77',
                 'scene0048_01 81', 'scene0048_01 90', 'scene0048_01 101', 'scene0048_01 94', 'scene0048_01 44',
                 'scene0048_01 8', 'scene0048_01 95', 'scene0048_01 32', 'scene0048_01 21', 'scene0048_01 131',
                 'scene0048_01 112', 'scene0048_01 11', 'scene0048_01 127', 'scene0048_01 23', 'scene0048_01 98',
                 'scene0048_01 104', 'scene0048_01 63', 'scene0048_01 111', 'scene0048_01 128', 'scene0048_01 52',
                 'scene0048_01 49', 'scene0048_01 72', 'scene0048_01 96', 'scene0048_01 79', 'scene0048_01 103',
                 'scene0048_01 24', 'scene0048_01 22', 'scene0048_01 42', 'scene0048_01 40', 'scene0048_01 119']
    width, height = 384, 288
    frame_idxs = [0, -1, 1]
    num_scales = 4
    is_train = True

    dataset = ScanNetDataset(data_path=data_path, filenames=filenames,
                             height=height, width=width,
                             frame_idxs=frame_idxs, num_scales=num_scales,
                             is_train=is_train)

    data = dataset[0]
    print()