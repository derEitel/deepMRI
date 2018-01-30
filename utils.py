import os
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import torch
from torch.nn.modules.module import _addindent


def load_nifti(file_path, z_factor=None, dtype=None):
    if dtype is None:
        dt = np.float32  
    else:
        dt = dtype
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dt)
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)
    return struct_arr

def load_masked_nifti(file_path, mask, sess, scan_pl, mask_pl):
    import tensorflow as tf
    struct_arr = load_nifti(file_path)
    struct_arr[np.where(np.isnan(struct_arr))] = 0

    c = tf.multiply(scan_pl, mask_pl)
    res = sess.run(c, feed_dict={scan_pl: struct_arr,
                                mask_pl: mask})
    return res

def normalize_float(x, min=-1):
    '''
     Function to normalize a matrix of floats.
     Can also deal with Pytorch dictionaries where the data matrix
     key is 'image'.
    '''
    was_dict = False
    if type(x) == dict:
        was_dict = True
        mat = x['image']
    else:
        mat = x
        mat = torch.from_numpy(mat)
    if min == -1:
        norm = 2 * (mat - torch.min(mat)) / (torch.max(mat) - torch.min(mat)) - 1
    elif min == 0:
        norm = (mat - torch.min(mat)) / (torch.max(mat) - torch.min(mat))
    if was_dict:
        x['image'] = norm
        return x
    else:
        return norm

def preprocess_img(x, mean, std):
    x = (x - mean) / (std + e)
    return x

def deprocess_img(x, mean, std):
    x = x * std + mean
    return x

class Normalize(object):
    """Normalize tensor with first and second moments."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor, eps=1e-12):
        tensor['image'] = (tensor['image'] - self.mean) / (self.std + eps)
        return tensor
    
    def denormalize(self, tensor, eps=1e-12):
        tensor['image'] = tensor['image']  * (self.std + eps) + self.mean
        return tensor

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights.
    Taken from wassname on Stackoverflow: https://stackoverflow.com/a/45528544
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr