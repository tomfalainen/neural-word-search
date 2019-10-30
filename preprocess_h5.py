#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:47:42 2016

@author: tomas
"""
import string
import os
import argparse
import copy
import json
from Queue import Queue
from threading import Thread, Lock

import h5py
import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import skimage.filters as fi
from scipy.misc import imresize

import misc.dataset_loader as dl
from misc.embeddings import dct, phoc
import misc.utils as utils
import misc.box_utils as box_utils

def extract_dtp(datum):
    img = imread(datum['id'])
    assert img.ndim == 2
    m = img.mean()
    if datum['id'].lower().find('iam') > -1 or datum['id'].lower().find('iiit_hws') > -1:
        threshold_range = np.array([0.9]) * m
    elif datum['id'].lower().find('washington') > -1:
        threshold_range = np.arange(0.7, 1.01, 0.1) * m
    else: #for botany and konzilsprotokolle
        threshold_range = np.arange(0.6, 0.91, 0.1) * m
        
    C_range=range(1, 50, 5) #horizontal range
    R_range=range(1, 50, 5) #vertical range
    region_proposals = dl.find_regions(img, threshold_range, C_range, R_range) 
    region_proposals, _ = utils.unique_boxes(region_proposals)
    datum['region_proposals'] = region_proposals.tolist()

def mt_extract_dtp(data):
    q = Queue()
    lock = Lock()

    for i, datum in enumerate(data):
        q.put((i, datum))
    
        def worker():
            while True:
              i, datum = q.get()
              extract_dtp(datum)
              lock.acquire()
              if i % 200 == 0:
                  print i
              lock.release()
              q.task_done()
          
    num_workers = 8
    for i in xrange(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()

def full_page_augment(data, outdir, tparams, num_images=2500, augment=True, reset=False):
    output_json = os.path.join(outdir, 'fullpage_augment/data.json')
    if not os.path.exists(output_json) or reset:
        train_data = [datum for datum in data if datum['split'] == 'train']
        vocab = utils.build_vocab(train_data) #vocab local to this function
        vocab_size = len(vocab)
        wtoi = {w:i for i, w in enumerate(vocab)}
                       
        od = os.path.join(outdir, 'fullpage_augment')
        if not os.path.exists(od):
            os.makedirs(od)
        
        words_by_label = [[] for i in range(vocab_size)] 
        shapes = []
        medians = []
        for datum in train_data:
            img = imread(datum['id'])
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
                
            medians.append(np.median(img))
            shapes.append(img.shape)
            for r in datum['regions']:
                x1, y1, x2, y2 = r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']
                word = img[y1:y2, x1:x2]
                label = r['label']
                ind = wtoi[label]
                words_by_label[ind].append(word)
        
        m = int(np.median(medians))
        augmented = []
        nwords = 256    
        s = 3           #inter word space
        box_id = 0
        for i in range(num_images):
            x, y = s, s #Upper left corner of box
            gt_boxes = []
            gt_labels = []
            shape = shapes[i % len(shapes)]
            canvas = create_background(m + np.random.randint(0, 20) - 10, shape)
            maxy = 0
            f = os.path.join(od, '%d.png' % i)
            regions = []
            for j in range(nwords):
                ind = np.random.randint(vocab_size) 
                k = len(words_by_label[ind])
                word = words_by_label[ind][np.random.randint(k)]

                #randomly transform word and place on canvas
                if augment:
                    try:
                        tword = utils.augment(word, tparams)
                    except:
                        tword = word
                else:
                    tword = word
                    
                h, w = tword.shape
                if x + w >= shape[1]: #done with row?
                    x = s
                    y = maxy + s
                    
                if y + h >= shape[0]: #done with page?
                    break
                
                #can happen for botany
                if tword.shape[0] > canvas.shape[0] or tword.shape[1] > canvas.shape[1]: 
                    continue
                
                x1, y1, x2, y2 = x, y, x + w, y + h
                canvas[y1:y2, x1:x2] = tword
                b = [x1, y1, x2, y2]
                gt_labels.append(vocab[ind])
                gt_boxes.append(b)
                x = x2 + s
                maxy = max(maxy, y2)
                r = {}
                r['id'] = box_id
                r['image'] = f
                r['height'] = b[3] - b[1]
                r['width'] = b[2] - b[0]
                r['label'] = vocab[ind]
                r['x'] = b[0]
                r['y'] = b[1]
                box_id += 1
                regions.append(r)

            imsave(f, canvas)
            d = {}
            d['gt_boxes'] = gt_boxes
            d['id'] = f
            d['split'] = 'train'
            d['regions'] = regions
            d['augmentation_type'] = 'full'
            augmented.append(d)
            
        #Multithreaded extraction of DTP proposals
        mt_extract_dtp(augmented)
        
        with open(output_json, 'w') as f:
            json.dump(augmented, f)
            
    else:
        with open(output_json) as f:
            augmented = json.load(f)
            
    for d in augmented:
        d['region_proposals'] = np.array(d['region_proposals'], dtype=np.int32)
        
    return augmented
            
def create_background(m, shape, fstd=2, bstd=10):
    canvas = np.ones(shape) * m
    noise = np.random.randn(shape[0], shape[1]) * bstd
    noise = fi.gaussian(noise, fstd)     #low-pass filter noise
    canvas += noise
    canvas = np.round(canvas)
    canvas = np.minimum(canvas, 255)
    canvas = canvas.astype(np.uint8)
    return canvas
   
def inplace_augment(data, outdir, tparams, num_images=2500, fold=1, reset=False):
    output_json = os.path.join(outdir, 'inplace_augment/data.json')
    if not os.path.exists(output_json) or reset:
        od = os.path.join(outdir, 'inplace_augment')
        if not os.path.exists(od):
            os.makedirs(od)
        
        augmented = []
        train_data = [datum for datum in data if datum['split'] == 'train']
        
        for i in xrange(num_images):
            datum = train_data[np.random.randint(len(train_data))]
            new_datum = copy.deepcopy(datum)
            new_datum['augmentation_type'] = 'inplace'
            path, f = os.path.split(new_datum['id'])
            img = imread(new_datum['id'])
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
                
            out = img.copy()
            boxes = new_datum['gt_boxes']
            for jj, b in enumerate(reversed(boxes)):
                #Some random values for weird boxes give value errors, just handle and ignore
                try: 
                    b = utils.close_crop_box(img, b)
                    word = img[b[1]:b[3], b[0]:b[2]]
                    aug = utils.augment(word, tparams)
                except ValueError:
                    continue
                    
                out[b[1]:b[3], b[0]:b[2]] = aug
            
            new_path = os.path.join(od, f[:-4] + '_%d.png' % i)
            imsave(new_path, out)
            new_datum['id'] = new_path
            augmented.append(new_datum)

        #Multithreaded extraction of DTP proposals
        mt_extract_dtp(augmented)

        with open(output_json, 'w') as f:
            json.dump(augmented, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            augmented = json.load(f) 
            
    for d in augmented:
        d['region_proposals'] = np.array(d['region_proposals'], dtype=np.int32)
            
    return augmented

def build_vocab_dict(vocab):
  token_to_idx, idx_to_token = {}, {}
  next_idx = 0
  for token in vocab:
    token_to_idx[token] = next_idx
    idx_to_token[next_idx] = token
    next_idx = next_idx + 1
    
  return token_to_idx, idx_to_token

def encode_word_embeddings(data, wtoe):
    we = []
    for datum in data:
        for r in datum['regions']:
            we.append(wtoe[r['label']])
            
    return np.array(we)
    
def encode_labels(data, wtoi):
    labels = []
    for datum in data:
        for r in datum['regions']:
            labels.append(wtoi[r['label']])
            
    return np.array(labels)
    
def encode_boxes(data, original_heights, original_widths, image_size, max_image_size,
                 box_type='gt_boxes', pad_proposals=False):
    all_boxes = []
    for i, datum in enumerate(data):
        H, W = original_heights[i], original_widths[i]
        scale = float(image_size) / max(H, W)

        if i % 1000 == 0:
            print "%s %d" % (box_type, i)
     
       #Needed for not so tightly labeled datasets, like washington
        if box_type == 'region_proposals' and pad_proposals:
            datum[box_type] = utils.pad_proposals(datum[box_type], (H, W), 10)
        
        boxes = np.array(datum[box_type])
        scaled_boxes = torch.from_numpy(np.round(scale * (boxes + 1) - 1))
        all_boxes.append(box_utils.x1y1x2y2_to_xcycwh(scaled_boxes).numpy())
    return np.vstack(all_boxes).astype(np.int32)
    
def build_img_idx_to_box_idxs(data, boxes='regions'):
  img_idx = 0
  box_idx = 0
  num_images = len(data)
  img_to_first_box = np.zeros(num_images, dtype=np.int32)
  img_to_last_box = np.zeros(num_images, dtype=np.int32)
  for datum in data:
    img_to_first_box[img_idx] = box_idx
    for region in datum[boxes]:
      box_idx += 1
    img_to_last_box[img_idx] = box_idx
    img_idx += 1
  
  return img_to_first_box, img_to_last_box

def add_images(data, h5_file, image_size, max_image_size, num_workers=5):
  num_images = len(data)
  shape = (num_images, 1, max_image_size[0], max_image_size[1])
  image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
  original_heights = np.zeros(num_images, dtype=np.int32)
  original_widths = np.zeros(num_images, dtype=np.int32)
  image_heights = np.zeros(num_images, dtype=np.int32)
  image_widths = np.zeros(num_images, dtype=np.int32)
  
  lock = Lock()
  q = Queue()
  
  for i, img in enumerate(data):
    q.put((i, img['id']))
    
  def worker():
    while True:
      i, filename = q.get()
      img = imread(filename)
      if img.ndim == 3:
          img = img_as_ubyte(rgb2gray(img))
      H0, W0 = img.shape[0], img.shape[1]
      img = imresize(img, float(image_size) / max(H0, W0))
      H, W = img.shape[0], img.shape[1]
      img = np.invert(img)

      lock.acquire()
      if i % 1000 == 0:
        print 'Writing image %d / %d' % (i, len(data))
      original_heights[i] = H0
      original_widths[i] = W0
      image_heights[i] = H
      image_widths[i] = W
      image_dset[i, :, :H, :W] = img
      lock.release()
      q.task_done()
      
  print('adding images to hdf5.... (this might take a while)')
  for i in xrange(num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  h5_file.create_dataset('image_heights', data=image_heights)
  h5_file.create_dataset('image_widths', data=image_widths)
  h5_file.create_dataset('original_heights', data=original_heights)
  h5_file.create_dataset('original_widths', data=original_widths)

def encode_splits(data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'train': 0, 'val': 1, 'test': 2}
  return [lookup[datum['split']] for datum in data]

def encode_augmentation_type(data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'none': 0, 'inplace': 1, 'full': 2}
  return [lookup[datum['augmentation_type']] for datum in data]

def encode_embeddings(data, f, vocab, alphabet, phoc_levels, itow):
     # encode dct embeddings
    dct_wtoe = {w:dct(w, 3, alphabet) for w in vocab}
    dct_word_embeddings = encode_word_embeddings(data, dct_wtoe)
    f.create_dataset('dct_word_embeddings', data=dct_word_embeddings)
    dct_itoe =  np.zeros((len(vocab), dct_word_embeddings.shape[1]))
    for ii in range(1, len(vocab)):
        dct_itoe[ii] = dct_wtoe[itow[ii]]
    f.create_dataset('dct_itoe', data=dct_itoe)

    # encode phoc embeddings
    phoc_wtoe = {w:phoc(w, alphabet, phoc_levels) for w in vocab}
    phoc_word_embeddings = encode_word_embeddings(data, phoc_wtoe)
    f.create_dataset('phoc_word_embeddings', data=phoc_word_embeddings)
    phoc_itoe =  np.zeros((len(vocab), phoc_word_embeddings.shape[1]))
    for ii in range(1, len(vocab)):
        phoc_itoe[ii] = phoc_wtoe[itow[ii]]
    f.create_dataset('phoc_itoe', data=phoc_itoe)

def create_dataset(dataset, augment, fold=1, reset=False, suffix=''):
    root = 'data/%s/' % dataset
    num_workers = 5
    image_size = 1720
    alphabet = dl.default_alphabet
    if dataset == 'konzilsprotokolle':
        alphabet = '&' + string.digits + string.ascii_lowercase
    
    phoc_levels = range(1, 6)
    dataset_full = dataset + '_fold%d' % fold
    augmentation_directory = root + dataset_full + '/'
    h5_output = root + dataset_full
    json_output = root + dataset_full
    
    if suffix:
        h5_output += '_' + suffix
        json_output += '_' + suffix

    h5_output += '.h5'
    json_output += '.json'
        
    # read in the data
    data = getattr(dl, 'load_' + dataset)(fold)
    sizes = []
    means = []
    if dataset != 'iiit_hws':
        for datum in data:
            datum['augmentation_type'] = 'none'
            img = imread(datum['id'])
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
                
            if datum['split'] == 'train':
                means.append(np.invert(img).mean())
            sizes.append(img.shape)
            
        sizes = np.array(sizes)
        max_image_size = sizes.max(axis=0)
    
    else:
        for datum in data:
            img = imread(datum['id'])
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
                
            if datum['split'] == 'train':
                means.append(np.invert(img).mean())
                
    image_mean = np.mean(means)
    if augment:
        num_images = 5000
        tparams = {}
        tparams['shear'] = (-5, 30)
        tparams['order'] = 1            #bilinear
        tparams['selem_size'] = (3, 4)  #max size for square selem for erosion, dilation
        tparams['rotate'] = (0, 1)
        tparams['hpad'] = (0, 1)
        tparams['vpad'] = (0, 1)
        tparams['keep_size'] = True
        
        #the inplace data also contain the original data
        if dataset == 'botany' or dataset == 'konzilsprotokolle':
            tparams['shear'] = (-5, 20)
            tparams['selem_size'] = (3, 3)
            
        inplace_data = inplace_augment(data, augmentation_directory, tparams, num_images=num_images/2, reset=reset)

        tparams['shear'] = (-5, 30)
        tparams['rotate'] = (-5, 5)
#        tparams['hpad'] = (0, 12)
#        tparams['vpad'] = (0, 12)
        tparams['keep_size'] = False
        full_page_data = full_page_augment(data, augmentation_directory, tparams, num_images=num_images/2, reset=reset)
        data += inplace_data + full_page_data 
        
    # create the output hdf5 file handle
    f = h5py.File(h5_output, 'w')
    
#    # add several fields to the file: images, and the original/resized widths/heights
    add_images(data, f, image_size, max_image_size, num_workers)
    f.create_dataset('image_mean', data=np.array([image_mean]))
    
    # add split information
    split = encode_splits(data)
    f.create_dataset('split', data=split)
    
    # add augmentation_type information
    augmentation_types = encode_augmentation_type(data)
    f.create_dataset('augmentation_types', data=augmentation_types)
    
#    # build vocabulary
    vocab = utils.build_vocab(data)
    wtoi, itow = build_vocab_dict(vocab) 
    
    #encode embeddings
    encode_embeddings(data, f, vocab, alphabet, phoc_levels, itow)
      
    # encode boxes
    original_heights = np.asarray(f['original_heights'])
    original_widths = np.asarray(f['original_widths'])
    gt_boxes = encode_boxes(data, original_heights, original_widths, image_size, max_image_size)
    f.create_dataset('boxes', data=gt_boxes)
    
    # write labels
    labels = encode_labels(data, wtoi)
    f.create_dataset('labels', data=labels)
    
    # integer mapping between image ids and region_proposals ids
    utils.filter_region_proposals(data, original_heights, original_widths, image_size)
    
    
    pad_proposals = dataset.find('washington') > -1 or dataset.find('botany') > -1  \
                    or dataset.find('konzilsprotokolle') > -1
                    
    region_proposals = encode_boxes(data, original_heights, original_widths, 
                                    image_size, max_image_size, 'region_proposals', pad_proposals)
    f.create_dataset('region_proposals', data=region_proposals)
    
    img_to_first_rp, img_to_last_rp = build_img_idx_to_box_idxs(data, 'region_proposals')
    f.create_dataset('img_to_first_rp', data=img_to_first_rp)
    f.create_dataset('img_to_last_rp', data=img_to_last_rp)
    
    # integer mapping between image ids and box ids
    img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
    f.create_dataset('img_to_first_box', data=img_to_first_box)
    f.create_dataset('img_to_last_box', data=img_to_last_box)
    f.close()
    
    # and write the additional json file 
    json_struct = {'wtoi': wtoi,'itow': itow}
    with open(json_output, 'w') as f:
        json.dump(json_struct, f)
        
    print 'finished creating %s' % dataset_full

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create h5 datasets')
    parser.add_argument('-dataset', help='The dataset used for training, one of "washington", "iam", "washington_small", or "iiit_hws"')
    parser.add_argument('-augment', help='use augmentation?', type=int)
    parser.add_argument('-suffix', default='', help='an optional suffix to h5 files')
    parser.add_argument('-reset', default=0, help='whether to redo augmentation', type=int)
    parser.add_argument('-folds', default=1, help='use multiple folds for washington', type=int)
    args = parser.parse_args()

    # if dataset is washington, do 4 folds
    if args.folds and args.dataset.find('washington') > -1:
        for fold in range(1, 5):
            create_dataset(args.dataset, suffix=args.suffix, augment=args.augment, 
                           fold=fold, reset=args.reset)
            
    else:
        create_dataset(args.dataset, suffix=args.suffix, augment=args.augment, 
                   fold=1, reset=args.reset)

