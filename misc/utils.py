#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:57:59 2017

@author: tomas
"""

import numpy as np
import easydict

from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import skimage.filters as fi
import skimage.transform as tf
import skimage.morphology as mor
from skimage.io import imread
np.errstate(divide='ignore', invalid='ignore')

def copy_log(rt, rf=None):
    for k, v in rt['log'].iteritems():
        rt[k] = v
    if rf:
        for k, v in rf['log'].iteritems():
            rf[k] = v

def average_dictionary(dicts, keys, add_statistics=False, scale=False):
    out = {}
    s = 1
    if scale:
        s = 100
        
    for key in keys:      
        vals = [d[key] * s for d in dicts]
        val = np.mean(vals)
        out[key] = val
        if add_statistics:
            out[key + '_std'] = np.std(vals)
            out[key + '_min'] = np.min(vals)
            out[key + '_max'] = np.max(vals)
        
    return easydict.EasyDict(out)

def getopt(opt, key, default=None):
    if opt.has_key(key):
        return opt[key]
    elif default != None:
        return default
    else:
        raise ValueError ("No default value provided for key %s" % key)
        
def ensureopt(opt, key):
        assert opt.has_key(key)

def replace_tokens(text, tokens):
    for t in tokens:
        text = text.replace(t, '')
        
    return text
    

def unique_boxes(boxes):
    tmp = np.array(boxes)
    ncols = tmp.shape[1]
    dtype = tmp.dtype.descr * ncols
    struct = tmp.view(dtype)
    uniq, index = np.unique(struct, return_index=True)
    tmp = uniq.view(tmp.dtype).reshape(-1, ncols)
    return tmp, index

    
def filter_region_proposals(data, original_heights, original_widths, image_size):
    """
    Remove duplicate region proposals when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.
    """
    for i, datum in enumerate(data):
        H, W = original_heights[i], original_widths[i]
        scale = float(image_size) / max(H, W)
        
        #Since we downsample the image 8 times before the roi-pooling, 
        #divide scaling by 8. Hardcoded per network architecture.
        scale /= 8
        okay = []
        for box in datum['region_proposals']:
            x, y = box[0], box[1]
            w, h = box[2] - x, box[3] - y
            x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
            w, h = round(scale*w), round(scale*h)  
            
            if x < 1: x = 1
            if y < 1: y = 1

            if w > 0 and h > 0:
                okay.append(box)
            
        #Only keep unique proposals in downsampled coordinate system, i.e., remove aliases 
        region_proposals, _ = unique_boxes(np.array(okay))
        datum['region_proposals'] = region_proposals
        
def filter_ground_truth_boxes(data, image_size=1720):
    """
    Remove too small ground truth boxes when downsampled to the roi-pooling size
    First it's the image scaling preprocessing then it's the downsampling in 
    the network.
    """
    for i, datum in enumerate(data):
        img = imread(datum['id'])
        H, W = img.shape
        scale = float(image_size) / max(H, W)
        
        #Since we downsample the image 8 times before the roi-pooling, 
        #divide scaling by 8. Hardcoded per network architecture.
        scale /= 8
        okay = []
        okay_gt = []
        assert(len(datum['regions']) == len(datum['gt_boxes']))
        for r, gt in zip(datum['regions'], datum['gt_boxes']):
            
            x, y, w, h = r['x'], r['y'], r['width'], r['height']
            xb, yb, wb, hb = gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]
            assert(xb == x)
            assert(yb == y)
            assert(wb == w)
            assert(hb == h)
            x, y = round(scale*(x-1)+1), round(scale*(y-1)+1)
            w, h = round(scale*w), round(scale*h)  
            
            if w > 1 and h > 1 and x > 0 and y > 0:
                okay.append(r)
                okay_gt.append(gt)
                
        datum['regions'] = okay
        datum['gt_boxes'] = okay_gt


def build_vocab(data):
    """ Builds a set that contains the vocab."""
    texts = []
    for datum in data:
        for r in datum['regions']:
            texts.append(r['label'])
  
    vocab, indeces = np.unique(texts, return_index=True)
    return vocab#, indeces

def pad_proposals(proposals, im_shape, pad=10):
    props = []
    for p in proposals:
        pp = [max(0, p[0] - pad), max(0, p[1] - pad), min(im_shape[1], p[2] + pad), min(im_shape[0], p[3] + pad)]
        props.append(pp)
    return np.array(props)

def close_crop_box(img, box):
    gray = rgb2gray(img[box[1]:box[3], box[0]:box[2]])
    t_img = gray < fi.threshold_otsu(gray)
    v_proj = t_img.sum(axis=1)
    h_proj = t_img.sum(axis=0)
    y1o = box[1] + max(v_proj.nonzero()[0][0] - 1, 0)
    x1o = box[0] + max(h_proj.nonzero()[0][0] - 1, 0)
    y2o = box[3] - max(v_proj.shape[0] - v_proj.nonzero()[0].max() - 1, 0)
    x2o = box[2] - max(h_proj.shape[0] - h_proj.nonzero()[0].max() - 1, 0)
    obox = (x1o, y1o, x2o, y2o)
    return obox

def outer_box(boxes):
    """
    Returns the bounding box of an (Nx4) array of boxes on form [x1, y1, x2, y2]
    """
    return np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])

### Augmentation stuff

def morph(img, tparams):
    ops = [mor.grey.erosion, mor.grey.dilation]
    t = np.random.randint(2)
    if t == 0:    
        selem = mor.square(np.random.randint(1, tparams['selem_size'][0]))
    else:
        selem = mor.square(np.random.randint(1, tparams['selem_size'][1]))
    return ops[t](img, selem)  
    
def affine(img, tparams):
    phi = (np.random.uniform(tparams['shear'][0], tparams['shear'][1])/180) * np.pi
    theta = (np.random.uniform(tparams['rotate'][0], tparams['rotate'][1])/180) * np.pi
    t = tf.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = tf.warp(img, t, order=tparams['order'], mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp

def close_crop(img, tparams):
    t_img = img < fi.threshold_otsu(img)
    nz = t_img.nonzero()
    pad = np.random.randint(low = tparams['hpad'][0], high = tparams['hpad'][1], size=2)    
    vpad = np.random.randint(low = tparams['vpad'][0], high = tparams['vpad'][1], size=2)    
    b = [max(nz[1].min() - pad[0], 0), max(nz[0].min() - vpad[0], 0), 
         min(nz[1].max() + pad[1], img.shape[1]), min(nz[0].max() + vpad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]

def augment(word, tparams):
    assert(word.ndim == 2)
    t = np.zeros_like(word)
    s = np.array(word.shape) - 1
    t[0, :] = word[0, :]
    t[:, 0] = word[:, 0]
    t[s[0], :] = word[s[0], :]
    t[:, s[1]] = word[:, s[1]]
    pad = int(np.median(t[t > 0]))

    tmp = np.ones((word.shape[0] + 8, word.shape[1] + 8), dtype = word.dtype) * pad
    tmp[4:-4, 4:-4] = word
    out = tmp
    out = affine(out, tparams)
    out = close_crop(out, tparams)        
    out = morph(out, tparams)
    if tparams['keep_size']:
        out = tf.resize(out, word.shape)
    out = img_as_ubyte(out)
    return out
