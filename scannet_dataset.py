import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(self, data_path, filenames, height, width,
                 frame_idxs, num_scales, is_train=False, img_ext='.jpg', *args, **kwargs):
        super(ScanNetDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1296, 968)
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.to_tensor = transforms.ToTensor()

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def __getitem__(self, index):
        inputs = {}

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1])

        inputs["filepath"] = self.filenames[index]

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i)
            inputs[("pose", i)] = torch.from_numpy(self.get_pose(folder, frame_index + i))

        K = self.get_K(folder, np.array(inputs[("color", 0, -1)]).shape)
        for scale in range(self.num_scales):
            K = K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

            for k in list(inputs):
                if "color" in k:
                    n, im, i = k
                    for i in range(self.num_scales):
                        inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        depth_gt = self.get_depth(folder, frame_index)
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

    def get_image_path(self, folder, frame_index):
        image_path = os.path.join(
            self.data_path,
            "imgs",
            folder,
            "{}.png".format(frame_index))
        return image_path

    def get_color(self, folder, frame_index):
        img_path = self.get_image_path(folder, frame_index)
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_pose(self, folder, frame_index):
        pose_path = os.path.join(
            self.data_path,
            "poses",
            folder,
            "{}.txt".format(frame_index))

        pose = np.loadtxt(pose_path).astype(np.float32)

        return pose

    def get_depth(self, folder, frame_index):
        depth_path = os.path.join(
            self.data_path,
            "depths",
            folder,
            "{}.npy".format(frame_index))

        depth_gt = np.load(depth_path)
        depth_gt = (depth_gt * 255.0 / np.max(depth_gt)).astype(np.uint8)
        depth_gt = Image.fromarray(depth_gt, 'L').resize(self.full_res_shape, Image.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        return depth_gt