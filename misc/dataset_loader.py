#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:10:04 2016

@author: tomas
"""
import os
import glob
import string
import json
from Queue import Queue
from threading import Thread, Lock

import scipy as sp
import scipy.io
import numpy as np
from skimage.io import imread, imsave
import cv2

import utils

default_alphabet = string.ascii_lowercase + string.digits

def extract_regions(t_img, C_range, R_range):
    """
    Extracts region propsals for a given image
    """
    all_boxes = []    
    for R in R_range:
        for C in C_range:
            s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
            n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
            all_boxes += boxes
                
    return all_boxes

    
def find_regions(img, threshold_range, C_range, R_range):
    """
    Extracts DTP from an image using different thresholds and morphology kernels    
    """

    ims = []
    for t in threshold_range:
        ims.append((img < t).astype(np.ubyte))
    
    ab = []
    for t_img in ims:
        ab += extract_regions(t_img, C_range, R_range)
        
    return ab

def extract_dtp(data, C_range, R_range, multiple_thresholds=True):
    lock = Lock()
    q = Queue()
    for i, datum in enumerate(data):
        q.put((i, datum['id']))
        
    def worker():
        while True:
            i, filename = q.get()
            proposal_file = filename[:-4] + '_dtp.npz'

            if i % 1000 == 0:
                lock.acquire()
                print 'Processing image %d / %d' % (i, len(data))
                lock.release()
            
            if not os.path.exists(proposal_file):
                try:
                    img = imread(filename)

                    #extract regions
                    m = img.mean()
                    if multiple_thresholds:
                        threshold_range = np.arange(0.7, 1.01, 0.1) * m
                    else:
                        threshold_range = np.array([0.9]) * m
                    region_proposals = find_regions(img, threshold_range, C_range, R_range) 
                    region_proposals, _ = utils.unique_boxes(region_proposals)
                    np.savez_compressed(proposal_file, region_proposals=region_proposals)
                except:
                    lock.acquire()
                    print 'exception thrown with file', filename
                    lock.release()

            q.task_done()
              
    num_workers = 6
    for i in xrange(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()
    
def load_washington_small(fold=1, root="data/washington/", alphabet=default_alphabet):
    data = load_washington(fold, root, alphabet)
    first = True #Keep the same validation page
    for datum in data:
        if datum['split'] == 'train':
            datum['split'] = 'test'
        elif datum['split'] == 'test':
            if first:
                first = False
                continue
            else:
                datum['split'] = 'train'
            
    return data

def load_washington(fold=1, root="data/washington/", alphabet=default_alphabet):
    output_json = root + 'washington_fold_%d.json' % fold
    if not os.path.exists(output_json):
        print "loading washington from files"
        files = sorted(glob.glob(os.path.join(root, 'gw_20p_wannot/*.tif')))
        gt_files = [f[:-4] + '_boxes.txt' for f in files]
        
        with open(os.path.join(root, 'gw_20p_wannot/annotations.txt')) as f:
            lines = f.readlines()

        texts = [l[:-1] for l in lines]        
        ntexts = [utils.replace_tokens(text.lower(), [t for t in text if t not in alphabet]) for text in texts]
        
        data = []
        ind = 0
        box_id = 0
        for i, (f, gtf) in enumerate(zip(files, gt_files)):
            with open(gtf, 'r') as ff:
                boxlines = ff.readlines()[1:]

            img = imread(f)
            h, w = img.shape
            gt_boxes = []
            for line in boxlines:
                tmp = line.split()
                x1 = int(float(tmp[0]) * w)
                x2 = int(float(tmp[1]) * w)
                y1 = int(float(tmp[2]) * h)
                y2 = int(float(tmp[3]) * h)
                box = (x1, y1, x2, y2)
                gt_boxes.append(box)
                
            labels = ntexts[ind:ind + len(gt_boxes)]
            ind += len(gt_boxes)
            labels = [unicode(l, errors='replace') for l in labels]
                      
            regions = []
            for p, l in zip(gt_boxes, labels):
                r = {}
                r['id'] = box_id
                r['image'] = f
                r['height'] = p[3] - p[1]
                r['width'] = p[2] - p[0]
                r['label'] = l
                r['x'] = p[0]
                r['y'] = p[1]
                regions.append(r)
                box_id += 1

            datum = {}
            datum['id'] = f
            datum['gt_boxes'] = gt_boxes
            datum['regions'] = regions
            data.append(datum)

        print "extracting DTP for washington" 
        C_range=range(1, 40, 3) #horizontal range
        R_range=range(1, 40, 3) #vertical range
        extract_dtp(data, C_range, R_range, multiple_thresholds=True)
        for datum in data:        
            proposals = np.load(datum['id'][:-4] + '_dtp.npz')['region_proposals']
            datum['region_proposals'] = proposals.tolist()
            
        print "extracting DTP done" 
        
        inds = np.squeeze(np.load(root + 'indeces.npy').item()['inds'])
        data = [data[i] for i in inds]
        
        #Train/val/test on different partitions based on which fold we're using
        data = np.roll(data, 5 * (fold - 1)).tolist()

        for j, datum in enumerate(data):
            if j < 14:
                datum['split'] = 'train'
            elif j == 14:
                datum['split'] = 'val'
            else:
                datum['split'] = 'test'
         
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    return data  

def load_iiit_hws(fold=1, root='data/iiit-hws/', vocab_size='10k', alphabet=default_alphabet):
    output_json = 'iiit_hws_%s.json' % vocab_size
    if not os.path.exists(output_json):    
        print "loading iiit_hws from files"
        dic = sp.io.loadmat(os.path.join(root, 'groundtruth/%s.mat' % vocab_size))
        
        texts = np.squeeze(dic['texts'])       #vocabulary
        vocab = [t[0] for t in texts]
        labels = np.squeeze(dic['labels']) - 1 # -1 for matlab conversion
        names = np.squeeze(dic['names'])       #file names
        t_inds = np.squeeze(dic['training_indeces']).astype(np.int32) - 1   # -1 for matlab conversion
        v_inds = np.squeeze(dic['validation_indeces']).astype(np.int32) - 1 # -1 for matlab conversion

        splits = np.zeros(len(labels), dtype=np.int32)
        splits[t_inds] = 1
        splits[v_inds] = 2
        data = []
        for i, (n, l) in enumerate(zip(names, labels)):
            if i > 0 and i % 1000000 == 0:
                print "loading iiit-hws dataset index %d out of %d" % (i, len(labels))
            path = os.path.join('Images_90K_Normalized', n[0])
            f = os.path.join(root, path)
            datum = {}
            datum['file'] = f
            datum['text'] = vocab[int(l)]
            datum['label'] = int(l)
            if splits[i] == 1:
                datum['split'] = 'train'
            elif splits[i] == 2:
                datum['split'] = 'val'
            else:
                datum['split'] = 'test'

            data.append(datum)
    
        inds = np.random.permutation(len(data))
        data = [data[i] for i in inds]
            
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    return data    

def load_iam(fold=1, root="data/iam/", nval=10, alphabet=default_alphabet):
    output_json = root + 'iam.json'
    if not os.path.exists(output_json): 
        print "loading iam from files"
        with open(os.path.join(root, 'words.txt')) as f:
            lines = f.readlines()
        
        lines = lines[18:] #Remove some description in beginning of file
        
        form2box = {}
        for line in lines:
            ls = line.split()
            f = '-'.join(ls[0].split('-')[:2])
            if not form2box.has_key(f):
                form2box[f] = []
            form2box[f].append(line)

        splits = {}
        with open(os.path.join(root, 'trainset.txt')) as f:
            trainset = f.readlines()
        splits['train'] = sorted(list(set(['-'.join(l[:-1].split('-')[:-1]) for l in trainset])))

        with open(os.path.join(root, 'validationset1.txt')) as f:
            valset = f.readlines()                
        
        with open(os.path.join(root, 'validationset2.txt')) as f:
            valset += f.readlines()                
            
        splits['val'] = sorted(list(set(['-'.join(l[:-1].split('-')[:-1]) for l in valset])))
        with open(os.path.join(root, 'testset.txt')) as f:
            testset = f.readlines()      
        splits['test'] = sorted(list(set(['-'.join(l[:-1].split('-')[:-1]) for l in testset])))

        os.makedirs(root + 'cropped_forms/')

        files = sorted(glob.glob(root + 'forms/*.png'))
        data = []
        box_id = 0
        for i, f in enumerate(files):
            gt_boxes = []
            regions = []
            form = f.split('/')[-1][:-4]
            lines = form2box[form]
            for line in lines:
                ls = line.split()
                status = ls[1]
                label = ls[-1]
                x, y, w, h = int(ls[3]), int(ls[4]), int(ls[5]), int(ls[6])
                b = [x, y, x + w, y + h]
                r = {}
                r['id'] = box_id
                r['image'] = form
                r['height'] = h
                r['width'] = w
                r['label'] = label
                r['x'] = x
                r['y'] = y
                r['status'] = status
                regions.append(r)
                gt_boxes.append(b)
                box_id += 1
            
            if len(gt_boxes) == 0:
                print 'fail', f
                
            datum = {}
            fs = f.split('/')[-1][:-4]
            if fs in splits['train']:
                datum['split'] = 'train'
            elif fs in splits['val']:
                datum['split'] = 'val'  
            elif fs in splits['test']:
                datum['split'] = 'test'  
            else:
                continue

            #crop form to relevant parts only and save
            img = imread(f)
            
            #calculate convex hull of all boxes, and add some margin height wise.
            ob = utils.outer_box(np.array(gt_boxes))
            margin = 50
            ob[1] = max(ob[1] - margin, 0) 
            ob[3] = min(ob[3] + margin, img.shape[0])
            od = root + 'cropped_forms/'
            fs = f.split('/')[-1]
            img = img[ob[1]:ob[3], :]
            file_id = od + fs
            imsave(od + fs, img)
            
            gt_boxes = np.array(gt_boxes)
            gt_boxes[:, 1] -= ob[1]
            gt_boxes[:, 3] -= ob[1]
            gt_boxes = gt_boxes.tolist()
            for r in regions:
                r['y'] -= ob[1]
            
            datum['gt_boxes'] = gt_boxes
            datum['regions'] = regions
            datum['id'] = file_id
            data.append(datum)
   
        print "extracting DTP for iam"
#        C_range=range(1, 40, 3) #horizontal range
#        R_range=range(1, 40, 3) #vertical range
        C_range=range(3, 50, 5) #horizontal range
        R_range=range(3, 50, 5) #vertical range
        extract_dtp(data, C_range, R_range, multiple_thresholds=False)
        for datum in data:        
            proposals = np.load(datum['id'][:-4] + '_dtp.npz')['region_proposals']
            datum['region_proposals'] = proposals.tolist()

        print "extracting DTP done" 
        
        #Some ground truth boxes for IAM are smaller than 8 pixels in width or height
        #We remove these as they collapse to zero width when forwarding through model
        #They are typically things like commas or periods, due to automatic word level annotation
        utils.filter_ground_truth_boxes(data)
        
        inds = np.random.permutation(len(data))
        data = [data[i] for i in inds]
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    for datum in data:
        assert len(datum['regions']) == len(datum['gt_boxes'])
        for r, gtb in zip(datum['regions'], datum['gt_boxes']):
            r['label'] = utils.replace_tokens(r['label'].lower(), [c for c in r['label'] if c not in alphabet])

    #too many validation pages takes too long, so we reduce at a possible cost of 
    #selecting the suboptimal model during training. 
    val_inds = [j for j, d in enumerate(data) if d['split'] == 'val']
    val_inds_remove = val_inds[nval:]
    data = [d for j, d in enumerate(data) if j not in val_inds_remove]
    return data
