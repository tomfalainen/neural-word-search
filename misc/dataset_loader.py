#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:10:04 2016

@author: tomas
"""
import os
import glob
import copy
import string
import json
from xml.dom import minidom
import zipfile

from Queue import Queue
from threading import Thread, Lock

import scipy as sp
import scipy.io
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from skimage.io import imread, imsave
import cv2

import utils
from boxIoU import bbox_overlaps

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

def generate_region_proposals(f, threshold_factors, row_range, column_range):
    img = imread(f)
    if img.ndim == 3:
        img = img_as_ubyte(rgb2gray(img))
    
    m = img.mean()
    threshold_range = threshold_factors * m
    region_proposals = find_regions(img, threshold_range, column_range, row_range) 
    region_proposals, _ = utils.unique_boxes(region_proposals)
    return region_proposals.tolist()

def load_botany(fold=1, root='data/botany/', full=False, alphabet=default_alphabet):
#def load_botany(fold=1, root='/mnt/harddisk/tomas/storage/botany/', full=False, alphabet=default_alphabet):
    output_json = root + 'botany'
    if full:
        output_json += '_full'
    output_json += '.json'

    if not os.path.exists(output_json):    
        data = botany_konz_process(root, 'botany', full)
        inds = np.random.permutation(len(data))
        data = [data[i] for i in inds]
        nval = 5
        k = 0
        for datum in data:
            if datum['split'] == 'train':
                if k < nval:
                    datum['split'] = 'val'
                    k += 1
                else:
                    break
                
        assert nval == len([datum['id'] for datum in data if datum['split'] == 'val'])

        with open(output_json, 'w') as f:
            json.dump(data, f)
            
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
            
#    with open('/mnt/harddisk/tomas/storage/botany/botany.json') as fp:
#        data2 = json.load(fp)
#        
#    for d in data:
#        f = d['id'].split('/')[-1]
#        for d2 in data2:
#            if d2['id'].split('/')[-1] == f:
#                d['region_proposals'] = d2['region_proposals']
#                d['id'] = d2['id']
#                d['gt_boxes'] = d2['gt_boxes']
#                d['regions'] = d2['regions']
    
    t = []
    for d in data:
        for r in d['regions']:
            t.append(r['label'])
            
    v = sorted(list(set(' '.join(t))))
    token = v[-4] #'£'
    replace = 'e'
    for d in data:
        for r in d['regions']:
            a = r['label']
            r['original_label'] = a
            a = a.replace(token, replace)
            a = a.lower()
            r['label'] = a
            r['label'] = utils.replace_tokens(r['label'], [c for c in r['label'] if c not in alphabet])
            
    return data   

#def load_konzilsprotokolle(fold=1, root='/mnt/harddisk/tomas/storage/konzilsprotokolle/',
#                           full=False, normalize_text=True, alphabet=default_alphabet):
def load_konzilsprotokolle(fold=1, root='data/konzilsprotokolle/',
                           full=False, normalize_text=True, alphabet=default_alphabet):
    output_json = root + 'konzilsprotokolle'
    if full:
        output_json += '_full'
    output_json += '.json'
    
    if not os.path.exists(output_json):    
        data = botany_konz_process(root, 'konzilsprotokolle', full)
        inds = np.random.permutation(len(data))
        data = [data[i] for i in inds]
        nval = 5
        k = 0
        for datum in data:
            if datum['split'] == 'train':
                if k < nval:
                    datum['split'] = 'val'
                    k += 1
                else:
                    break
                
        assert nval == len([datum['id'] for datum in data if datum['split'] == 'val'])

        with open(output_json, 'w') as f:
            json.dump(data, f)
            
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    
    t = []
    for d in data:
        for r in d['regions']:
            t.append(r['label'])
            
    v = sorted(list(set(' '.join(t))))
    tokens = v[-8:] #'ßäéöü'
    replace = 'AOUbaeou'
    replace = ['A&', 'O&', 'U&', 'b', 'a&', 'e&', 'o&', 'u&']
    
    for d in data:
        for r in d['regions']:
            a = r['label']
            r['original_label'] = a
            for token, rep in zip(tokens, replace):
                a = a.replace(token, rep)
            for token in [' ', '"', '(', ')', '-', '.', ',', '/', ':', ';', '?', '_',  u'\xa7']:
                a = a.replace(token, '')
            a = a.lower()
            r['label'] = a
            
    return data

def botany_konz_process(root, dataset, full):
    ds = dataset.capitalize()
#    if not os.path.exists(root + '%s_Train_I_PageImages/' % ds):
    #unzip everything
    zfiles = ['%s_Train_I_PageImages.zip' % ds,
              '%s_Train_I_XML.zip' % ds,
              '%s_Train_II_PageImages.zip' % ds,
              '%s_Train_II_XML.zip' % ds,
              '%s_Train_III_PageImages.zip' % ds,
              '%s_Train_III_XML.zip' % ds,
              '%s_Test_GT.zip' % ds,
              '%s_Test_PageImages.zip' % ds,
              '%s_Test_QryImages.zip' % ds,
              '%s_Test_QryStrings.zip' % ds,
              ]
    for zf in zfiles:
        with zipfile.ZipFile(root + zf,"r") as zip_ref:
            zip_ref.extractall(root)
    
    #Train and val files
    files = sorted(glob.glob(root + '%s_Train_III_PageImages/*.jpg' % ds))
    if full:
        files += sorted(glob.glob(root + '%s_Train_II_PageImages/*.jpg' % ds))
        files += sorted(glob.glob(root + '%s_Train_I_PageImages/*.jpg' % ds))
    
    doc = minidom.parse(root + '%s_Train_III_WL.xml' % ds)
    words = doc.getElementsByTagName('spot')
    if full:
        doc = minidom.parse(root + '%s_Train_II_WL.xml' % ds)
        words += doc.getElementsByTagName('spot')
        doc = minidom.parse(root + '%s_Train_I_WL.xml' % ds)
        words += doc.getElementsByTagName('spot')
        
    threshold_factors = np.arange(0.7, 1.01, 0.1)
    column_range=range(3, 50, 4) #horizontal range
    row_range=range(3, 50, 4) #vertical range
        
#    threshold_factors = np.arange(0.4, 0.81, 0.1)
#    column_range=range(1, 50, 2) #horizontal range
#    row_range=range(1, 50, 2) #vertical range
    medians = []
    data = []
    box_id = 0
    for f in files:
        img = imread(f)
        if img.ndim == 3:
            img = img_as_ubyte(rgb2gray(img))
            
        #reduce size by 2 and save 
#        img = img_as_ubyte(rescale(img, 0.5))
        img = sp.misc.imresize(img, 0.5)
        imsave(f, img)
        medians.append(np.median(img))
        H, W = img.shape
        gt_boxes = []
        transcriptions = []
        for word in words:
            if word.getAttribute('image') == f.split('/')[-1]:
                x, y = int(word.getAttribute('x')), int(word.getAttribute('y'))
                w, h = int(word.getAttribute('w')), int(word.getAttribute('h'))
                b = [x, y, x + w, y + h]
                x1, y1, x2, y2 = b
                transcriptions.append(word.getAttribute('word'))
                gt_boxes.append(b)
                
        if len(gt_boxes) == 0:
            print 'No ground truth for %s, skipping' % f
            continue
        
        gt_boxes = np.round((np.array(gt_boxes) -1) * 0.5 + 1).astype(np.int32).tolist()
        
        regions = []
        for b, label in zip(gt_boxes, transcriptions):
            r = {}
            r['id'] = box_id
            r['image'] = f
            r['height'] = b[3] - b[1]
            r['width'] = b[2] - b[0]
            r['label'] = label
            r['x'] = b[0]
            r['y'] = b[1]
            box_id += 1
            regions.append(r)
            
        #extract region proposals
        region_proposals = generate_region_proposals(f, threshold_factors, column_range, row_range)
        
        d = {}
        d['gt_boxes'] = gt_boxes
        d['id'] = f
        d['split'] = 'train'
        d['regions'] = regions
        d['region_proposals'] = region_proposals
        data.append(d)

    #Test files
    files = sorted(glob.glob(root + '%s_Test_PageImages/*.jpg' % ds))
    doc = minidom.parse(root + '%s_Test_GT_SegFree_QbS.xml' % ds)
    words = doc.getElementsByTagName('spot')

    #% Test to compare to QbE -> same boxes
    #for a few images the sorting boxes appear to differ, but this is due to sorting
    #order when x1 is the same
    doc = minidom.parse(root + '%s_Test_GT_SegFree_QbE.xml' % ds)
    qbe_spots = doc.getElementsByTagName('spot')
    
    query_files = sorted(glob.glob(root + '%s_Test_QryImages/*.jpg' % ds))
    query_images = [img_as_ubyte(rgb2gray(imread(f))) for f in query_files]
    query_transcriptions = ['' for i in range(len(query_files))]

    test_images = []
    for f in files:
        img = imread(f)
        if img.ndim == 3:
            img = rgb2gray(img)
            
#        img = img_as_ubyte(rescale(img, 0.5))
        img = sp.misc.imresize(img, 0.5)
        imsave(f, img)
        test_images.append(img)
        gt_boxes = []
        transcriptions = []
        for word in words:
            if word.getAttribute('image') == f.split('/')[-1]:
                transcriptions.append(word.getAttribute('word'))
                x, y = int(word.getAttribute('x')), int(word.getAttribute('y'))
                w, h = int(word.getAttribute('w')), int(word.getAttribute('h'))
                b = [x, y, x + w, y + h]
                gt_boxes.append(b)
       
        qbe_boxes = []
        qbe_transcriptions = []
        for word in qbe_spots:
            if word.getAttribute('image') == f.split('/')[-1]:
                qbe_transcriptions.append(word.getAttribute('word'))
                x, y = int(word.getAttribute('x')), int(word.getAttribute('y'))
                w, h = int(word.getAttribute('w')), int(word.getAttribute('h'))
                b = [x, y, x + w, y + h]
                qbe_boxes.append(b)
    
        qbe_boxes, qbe_transcriptions = np.array(qbe_boxes), np.array(qbe_transcriptions)
        gt_boxes, transcriptions = np.array(gt_boxes), np.array(transcriptions)
        gt_boxes = np.round((np.array(gt_boxes) -1) * 0.5 + 1).astype(np.int32)
        qbe_boxes = np.round((np.array(qbe_boxes) -1) * 0.5 + 1).astype(np.int32)
    
        overlaps = bbox_overlaps(qbe_boxes, gt_boxes)
        amax = overlaps.argmax(axis=1)
        tmp = transcriptions[amax]
        assert overlaps.max(axis=1).sum() == len(qbe_boxes)
        assert np.all(overlaps.max(axis=1) == np.ones(len(qbe_boxes)))
        assert len(qbe_transcriptions) == len(tmp)
        for ff, t in zip(qbe_transcriptions, tmp):
            for k, qf in enumerate(query_files):
                if qf.split('/')[-1] == ff:
                    query_transcriptions[k] = t

#    for f in files:
#        gt_boxes = []
#        transcriptions = []
#        for word in words:
#            if word.getAttribute('image') == f.split('/')[-1]:
#                transcriptions.append(word.getAttribute('word'))
#                x, y = int(word.getAttribute('x')), int(word.getAttribute('y'))
#                w, h = int(word.getAttribute('w')), int(word.getAttribute('h'))
#                b = [x, y, x + w, y + h]
#                gt_boxes.append(b)
#        
#        gt_boxes = np.round((np.array(gt_boxes) -1) * 0.5 + 1).astype(np.int32).tolist()

        gt_boxes = gt_boxes.tolist()
        regions = []
        for b, label in zip(gt_boxes, transcriptions):
            r = {}
            r['id'] = box_id
            r['image'] = f
            r['height'] = b[3] - b[1]
            r['width'] = b[2] - b[0]
            r['label'] = label
            r['x'] = b[0]
            r['y'] = b[1]
            box_id += 1
            regions.append(r)
            
        #extract region proposals
        region_proposals = generate_region_proposals(f, threshold_factors, column_range, row_range)
    
        d = {}
        d['gt_boxes'] = gt_boxes
        d['id'] = f
        d['split'] = 'test'
        d['regions'] = regions
        d['region_proposals'] = region_proposals 
        data.append(d)

    for qf, qt, qi in zip(query_files, query_transcriptions, query_images):
        kk = 0
        candidate_words = []
        candidate_boxes = []
        cw2region = []
        region2datum = []
        for dc, datum in enumerate(data):
            if datum['split'] == 'test':
                img = test_images[kk]
                kk += 1
                for rc, r in enumerate(datum['regions']):
                    if r['label'] == qt:
                        b = [r['y'],r['x'],r['height'], r['width']]
                        wi = img[r['y']:r['y'] + r['height'], r['x']:r['x'] + r['width']]
                        candidate_boxes.append(b)
                        candidate_words.append(wi)
                        cw2region.append((rc, dc))
                        
                    region2datum.append(dc)
            
            
        #To find which GT region a query image belongs to, compare images
        pixels = []
        for cw in candidate_words:
            ccw = img_as_ubyte(resize(cw, qi.shape))
            c = np.abs(ccw.astype(np.float32) - qi.astype(np.float32))
            pixels.append(c.sum())
            
        rc, dc = cw2region[np.argmin(pixels)]
        region = data[dc]['regions'][rc]
        region['query_file'] = qf

    return data