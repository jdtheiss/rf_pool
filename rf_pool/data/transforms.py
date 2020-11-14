import re

import numpy as np
import random
from scipy import ndimage
import torch
from torchvision.transforms import *
from torch.nn import functional

def _get_doc(cls, fn=None, replace=['','']):
    if fn is None and hasattr(cls, '__doc__'):
        doc = getattr(cls, '__doc__')
    elif hasattr(cls, fn) and hasattr(getattr(cls, fn), '__doc__'):
        doc = getattr(getattr(cls, fn), '__doc__')
    else:
        return None
    if type(doc) is str:
        doc = doc.replace(*replace)
    return doc

def apply_transforms(*args, transforms=None):
    raise NotImplementedError

def rgb2gray(img):
    assert img.shape[1] == 3
    w = torch.tensor([0.2125, 0.7154, 0.0721])
    return torch.matmul(img.permute(0,2,3,1), w).unsqueeze(1)

class CenterCrop(CenterCrop):
    __doc__ = _get_doc(CenterCrop, replace=['PIL','Tensor'])
    def __call__(self, img):
        img_h, img_w = img.shape[-2:]
        c_h = img_h // 2
        c_w = img_w // 2
        s_h = max(c_h - self.size[0] // 2, 0)
        s_w = max(c_w - self.size[1] // 2, 0)
        e_h = min(s_h + self.size[0], img_h)
        e_w = min(s_w + self.size[1], img_w)
        new_shape = img.shape[:-2] + (e_h-s_h, e_w-s_w)
        with torch.no_grad():
            return img.flatten(0, -2)[s_h:e_h, s_w:e_w].reshape(new_shape)
    __call__.__doc__ = _get_doc(CenterCrop, fn='__call__', replace=['PIL','Tensor'])

class ColorJitter(ColorJitter):
    __doc__ = _get_doc(ColorJitter, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(ColorJitter, fn='__call__', replace=['PIL','Tensor'])

class FiveCrop(FiveCrop):
    __doc__ = _get_doc(FiveCrop, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(FiveCrop, fn='__call__', replace=['PIL','Tensor'])

class Grayscale(Grayscale):
    __doc__ = _get_doc(Grayscale, replace=['PIL','Tensor'])
    def __call__(self, img):
        requires_grad = img.requires_grad
        with torch.no_grad():
            img = rgb2gray(img)
        return img.requires_grad_(requires_grad)
    __call__.__doc__ = _get_doc(Grayscale, fn='__call__', replace=['PIL','Tensor'])

class Pad(Pad):
    __doc__ = _get_doc(Pad, replace=['PIL','Tensor'])
    def __init__(self, *args, **kwargs):
        super(Pad, self).__init__(*args, **kwargs)
        if not isinstance(self.padding, (tuple, list)):
            self.padding = (self.padding,) * 4

    def __call__(self, img):
        return functional.pad(img, self.padding, self.padding_mode, self.fill)
    __call__.__doc__ = _get_doc(Pad, fn='__call__', replace=['PIL','Tensor'])

class RandomAffine(RandomAffine):#TODO:test
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample (int, optional):
            Optional interpolation. See `ndimage.affine_transform` for more information.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.

    """
    def __init__(self, *args, **kwargs):
        super(RandomAffine, self).__init__(*args, **kwargs)
        if not isinstance(self.fillcolor, (tuple, list)):
            self.fillcolor = (self.fillcolor,)

    @staticmethod
    def get_params(degrees, translate, scale, shear, img_size):
        mat = np.eye(2)
        # scale
        if scale is not None:
            scale = (random.uniform(*scale),)*2
            mat = np.dot(np.diag(scale), mat)
        # rotation
        if degrees is not None:
            degrees = -random.uniform(*degrees)
            theta = degrees * np.pi / 180.
            rot_mat = np.eye(2)
            rot_mat[:2,:2] = np.array([[np.cos(theta),-np.sin(theta)],
                                       [np.sin(theta),np.cos(theta)]])
            mat = np.dot(rot_mat, mat)
        # translation
        if translate is not None:
            translate = (translate[1] * img_size[0],
                         translate[0] * img_size[1])
            translate = [random.uniform(-translate[0], translate[0]),
                         -random.uniform(-translate[1], translate[1])]
        else:
            translate = np.zeros(2,)
        # shear
        if shear is not None: #TODO: update based on degrees...
            shear = [random.uniform(*shear[-2:]), random.uniform(*shear[:2])]
            shear_mat = np.eye(2)
            shear_mat[:2,:2] = -np.flip(np.diag(shear), -1) + np.eye(2)
            mat = np.dot(shear_mat, mat)
        return np.linalg.inv(mat), translate

    def __call__(self, img):
        requires_grad = img.requires_grad
        img = img.detach()
        with torch.no_grad():
            # get transformation matrix
            img_shape = img.shape
            img_size = img_shape[-2:]
            mat, translate = self.get_params(self.degrees, self.translate,
                                             self.scale, self.shear, img_size)
            half_img = 0.5 * np.array(img_size)
            offset = half_img - half_img.dot(mat.T) - translate
            # apply affine
            img = img.reshape((-1, len(self.fillcolor)) + img_size).numpy()
            for i in range(img.shape[0]):
                for j, fill in enumerate(self.fillcolor):
                    img[i,j] = ndimage.affine_transform(img[i,j], mat,
                                                          offset=offset,
                                                          order=self.resample,
                                                          cval=fill)
            img = img.reshape(img_shape)
        return torch.tensor(img, requires_grad=requires_grad)
    __call__.__doc__ = _get_doc(RandomAffine, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomCrop(RandomCrop):
    __doc__ = _get_doc(RandomCrop, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(RandomCrop, fn='__call__', replace=['PIL','Tensor'])

class RandomGrayscale(RandomGrayscale):
    __doc__ = _get_doc(RandomGrayscale, replace=['PIL','Tensor'])
    def __call__(self, img):
        requires_grad = img.requires_grad
        if random.random() < self.p:
            with torch.no_grad():
                img = rgb2gray(img)
        return img.requires_grad_(requires_grad)
    __call__.__doc__ = _get_doc(RandomGrayscale, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomHorizontalFlip(RandomHorizontalFlip):
    __doc__ = _get_doc(RandomHorizontalFlip, replace=['PIL','Tensor'])
    def __call__(self, img):
        if random.random() < self.p:
            return torch.tensor(np.flip(img.detach().numpy(), -1),
                                requires_grad=img.requires_grad)
        return img
    __call__.__doc__ = _get_doc(RandomHorizontalFlip, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomPerspective(RandomPerspective):
    __doc__ = _get_doc(RandomPerspective, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(RandomPerspective, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomResizedCrop(RandomResizedCrop):
    __doc__ = _get_doc(RandomResizedCrop, replace=['PIL','Tensor'])
    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        raise NotImplementedError
    __call__.__doc__ = _get_doc(RandomResizedCrop, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomRotation(RandomRotation):
    __doc__ = _get_doc(RandomRotation, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(RandomRotation, fn='__call__',
                                replace=['PIL','Tensor'])

class RandomVerticalFlip(RandomVerticalFlip):
    __doc__ = _get_doc(RandomVerticalFlip, replace=['PIL','Tensor'])
    def __call__(self, img):
        if random.random() < self.p:
            return torch.tensor(np.flip(img.detach().numpy(), -2),
                                requires_grad=img.requires_grad)
        return img
    __call__.__doc__ = _get_doc(RandomVerticalFlip, fn='__call__',
                                replace=['PIL','Tensor'])

class Resize(Resize):
    __doc__ = _get_doc(Resize, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(Resize, fn='__call__', replace=['PIL','Tensor'])

class TenCrop(TenCrop):
    __doc__ = _get_doc(TenCrop, replace=['PIL','Tensor'])
    def __call__(self):
        raise NotImplementedError
    __call__.__doc__ = _get_doc(TenCrop, fn='__call__', replace=['PIL','Tensor'])
