#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:39:56 2017

@author: tomas
"""
from __future__ import print_function
import json
import h5py
import utils
import numpy as np
import embeddings as emb
import dataset_loader as dl
from datasets import Dataset

class H5Dataset(Dataset):
    def __init__(self, opt, split, root='data/'):
        if split == 0:
            self.dataset = utils.getopt(opt, 'dataset')
        else:
            self.dataset = utils.getopt(opt, 'val_dataset')
            
        root += '%s/' % self.dataset
            
        self.debug_max_train_images = utils.getopt(opt, 'debug_max_train_images', -1)
        self.embedding = utils.getopt(opt, 'embedding')
        self.fold = utils.getopt(opt, 'fold')
        self.image_size = opt.image_size
        self.split_num = split
        self.dtp_train = opt.dtp_train
        self.opt = opt
        num2split = {0:'train', 1:'val', 2:'test'}
        self.split = num2split[split]
        self.train = self.split == 'train'
        self.alphabet = dl.default_alphabet
        self.data = getattr(dl, 'load_%s' % self.dataset)(fold=self.fold, alphabet=self.alphabet)
        self.data_split = [d for d in self.data if d['split'] == self.split]
        self.ghosh = opt.ghosh

        self.h5_file = root + self.dataset + '_fold%d.h5' % self.fold
        self.json_file = root + self.dataset + '_fold%d.json' % self.fold
        
        if self.dataset != 'iiit_hws':
            self.split_vocab = utils.build_vocab(self.data_split)
        else:
            self.split_vocab = np.unique([d['label'] for d in self.data])
                
        if self.ghosh: 
            self.h5_file = root + 'washington_fold1_ghosh.h5'
            self.json_file = root + 'washington_fold1_ghosh.json'
        
        show = not opt.quiet
  
        #load the json file which contains additional information about the dataset
        if show:
            print('DataLoader loading json file: ', self.json_file)
        with open(self.json_file, 'r') as f:
            self.info = json.load(f)
            
        self.vocab_size = len(self.info['itow'])
        
        #Convert keys in idx_to_token from string to integer
        itow = {} 
        for k, v in self.info['itow'].iteritems():
            itow[int(k)-1] = v
        self.info['itow'] = itow
        
        self.itow = itow
        self.wtoi = {w:i for i, w in itow.iteritems()}
        
        self.resolution = 3
        #boils down to whether or not the all embeddings should match their label
        self.bins=len(self.alphabet) * 2
        self.ngrams=2
        self.unigram_levels = range(1, 6)
        self.emb_func = getattr(emb, opt.embedding)
        self.args = (self.resolution, self.alphabet)
        if opt.embedding == 'ngram_dct':
            self.args += (self.ngrams, self.bins)
        elif opt.embedding == 'phoc':
            self.args = (self.alphabet, self.unigram_levels)
            
        if opt.embedding_loss == 'phocnet':
            self.wtoe = {w:self.emb_func(w, *self.args) for i, w in self.itow.iteritems()} #word embedding table
        else:
            self.wtoe = {w:self.normalize(self.emb_func(w, *self.args)) for i, w in self.itow.iteritems()} #word embedding table
        
        self.iam = self.dataset == 'iam'
        if self.iam:
            with open('data/iam/stopwords.txt') as f:
                tmp = f.readline()[:-1]
            self.stopwords = tmp.split(',')
            
        if not self.train:
           self.init_queries()

        # open the hdf5 file
        if show:
            print('DataLoader loading h5 file: ', self.h5_file)
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.boxes = self.h5_file.get('boxes').value
        self.image_heights = self.h5_file.get('image_heights').value
        self.image_widths = self.h5_file.get('image_widths').value
        self.img_to_first_box = self.h5_file.get('img_to_first_box').value
        self.img_to_last_box = self.h5_file.get('img_to_last_box').value
        self.labels = self.h5_file.get('labels').value - 1
        self.word_embedding = self.h5_file.get(self.embedding + '_word_embeddings').value
        self.img_to_first_rp = self.h5_file.get('img_to_first_rp').value
        self.img_to_last_rp = self.h5_file.get('img_to_last_rp').value
        self.original_heights = self.h5_file.get('original_heights').value
        self.original_widths = self.h5_file.get('original_widths').value
        self.split_inds = self.h5_file.get('split').value
  
        #extract image size from dataset
        images_size = self.h5_file.get('images').shape
        assert len(images_size) == 4, '/images should be a 4D tensor'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_height = images_size[2]
        self.max_image_width = images_size[3]

        #extract some attributes from the data
        self.num_regions = self.boxes.shape[0]
        self.image_mean = self.h5_file.get('/image_mean').value[0]

        #set up index ranges for the different splits
        self.train_ix = []
        self.val_ix = []
        self.test_ix = []
        for i in range(self.num_images):
            if self.split_inds[i] == 0: self.train_ix.append(i)
            if self.split_inds[i] == 1: self.val_ix.append(i)
            if self.split_inds[i] == 2: self.test_ix.append(i)

        if show:
            print('assigned %d/%d/%d images to train/val/test.' % (len(self.train_ix), len(self.val_ix), len(self.test_ix)))
            print('initialized DataLoader:')
            print('#images: %d, #regions: %d' % (self.num_images, self.num_regions))

    def get_image_max_size(self):
        return self.max_image_height, self.max_image_width

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.info.itow
    
    def __getitem__(self, index):
        split = self.split_num
        if split == 0:
            iterate = False
        else:
            iterate = True

        assert split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)'
        
        if split == 0: split_ix = self.train_ix
        if split == 1: split_ix = self.val_ix
        if split == 2: split_ix = self.test_ix
        assert len(split_ix) > 0, 'split is empty?'

        # pick an index of the datapoint to load next
        #ri is iterator position in local coordinate system of split_ix for this split
        max_index = len(split_ix)
        if self.debug_max_train_images > 0: max_index = self.debug_max_train_images
        if iterate:
            ri = index % max_index

        else:
            #pick an index randomly
            ri = np.random.randint(max_index)
  
        ix = split_ix[ri]
        assert ix != None, 'bug: split ' + str(split) + ' was accessed out of bounds with ' + str(ri)
  
        #fetch the image
        img = self.h5_file.get('/images')[ix]

        # crop image to its original width/height, get rid of padding, and dummy first dim
        img = img[:, :self.image_heights[ix], :self.image_widths[ix]]
        img = img.astype(np.float32)        # convert to float
        img /= 255.0                        # convert to [0, 1]
        img -= (self.image_mean / 255.0)    # subtract mean

        # fetch the corresponding labels array
        r0 = self.img_to_first_box[ix] # - 1  #for python, start from zero, 
        r1 = self.img_to_last_box[ix] #Nothing needed here since lua = inclusive, python exclusive
  
        embeddings = self.word_embedding[r0:r1]
        box_batch = self.boxes[r0:r1].copy()
#        box_batch[:, :2] -= 1
        labels = self.labels[r0:r1]
        C, H, W = img.shape
    
        #batch the boxes and labels and embeddings
        assert box_batch.ndim == 2

        ow, oh = self.original_widths[ix], self.original_heights[ix]

        if split == 0:
            out = (img, box_batch, embeddings, labels)
            if self.dtp_train:
                r0 = self.img_to_first_rp[ix] #- 1 #Same as with img_to_first_box
                r1 = self.img_to_last_rp[ix]
                region_proposals = self.h5_file.get('/region_proposals')[r0:r1].copy()
#                region_proposals[:, :2] -= 1
                out += (region_proposals, )
                
        elif split == 1 or split == 2:
            r0 = self.img_to_first_rp[ix] - 1 #Same as with img_to_first_box
            r1 = self.img_to_last_rp[ix]
            region_proposals = self.h5_file.get('/region_proposals')[r0:r1].copy()
            region_proposals[:, :2] -= 1
            out = img, (oh, ow), box_batch, region_proposals, embeddings, labels
            
        return out

    def __len__(self):
        if self.split_num == 0:
            return len(self.train_ix)
        elif self.split_num == 1:
            return len(self.val_ix)
        elif self.split_num == 2:
            return len(self.test_ix)
