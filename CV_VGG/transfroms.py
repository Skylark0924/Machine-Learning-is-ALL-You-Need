import torch
import numpy as np
import PIL
import cv2
import random


# TODO: implementation transformations for task3;
# You cannot directly use them from pytorch, but you are free to use functions from cv2 and PIL
class Padding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img, **kwargs):
        old_width, old_height = img.size
        background_color = (255, 255, 255)

        new_width = old_width + 2 * self.padding
        new_height = old_height + 2 * self.padding
        new_img = np.full((new_width, new_height, 3), background_color, dtype=np.uint8)

        new_img[self.padding:self.padding + old_height, self.padding:self.padding + old_width] = img
        img = PIL.Image.fromarray(new_img)
        return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, **kwargs):
        w, h = img.size
        y1 = torch.randint(0, h - self.size + 1, size=(1,)).item()
        x1 = torch.randint(0, w - self.size + 1, size=(1,)).item()
        y2 = y1 + self.size
        x2 = x1 + self.size
        new_img = img.crop((x1, y1, x2, y2))
        return new_img


class RandomFlip(object):
    def __init__(self, ):
        pass

    def __call__(self, img, **kwargs):
        rand_num = torch.rand(1)
        if rand_num < 0.5:
            new_img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        elif rand_num >= 0.5:
            new_img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # new_img = img.rotate(rot_angle, resample=PIL.Image.BILINEAR, expand=False, fillcolor=(255, 0, 255))
        return new_img


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, **kwargs):
        _, w, h = img.shape
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)  # 返回随机数/数组(整数)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
