#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:01:39 2016

@author: tomas
"""

import string
import numpy as np
import scipy.fftpack as sf

def dct(s, resolution, alphabet):
    if len(s) == 0:
        return np.zeros(resolution * len(alphabet), dtype=np.float32)
    im = np.zeros([len(alphabet),len(s)], 'single')
    F = np.zeros([len(alphabet),len(s)], 'single')
    for jj in range(0,len(s)):
        c = s[jj]
        im[string.find(alphabet, c), jj] = 1.0

    for ii in range(0,len(alphabet)):
        F[ii,:] = sf.dct(im[ii,:])

    A = F[:,0:resolution]
    B = np.zeros([len(alphabet),max(0,resolution-len(s))])
    return np.hstack((A,B)).flatten().astype(np.float32)
    
def phoc(word, alphabet, unigram_levels, bigram_levels=None, phoc_bigrams=None):
    """    Created on Dec 17, 2015

    @author: ssudholt
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        alphabet (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels for the unigrams in PHOC
        phoc_bigrams (list of str): list of bigrams to be used in the PHOC
        phoc_bigram_levls (list of int): the levels of the bigrams in the PHOC

    Returns:
        the PHOC for the given word
    """    
    phoc_size = len(alphabet) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams)*np.sum(bigram_levels)
    phocs = np.zeros((phoc_size,), dtype=np.float32)
    if len(word) == 0:
        return phocs
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k+1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]
    
    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(alphabet)}
    
    # iterate through all the words
    n = len(word) 
    for index, char in enumerate(word):
        char_occ = occupancy(index, n)
        if char not in char_indices:
            raise ValueError()
        char_index = char_indices[char]
        for level in unigram_levels:
            for region in range(level):
                region_occ = occupancy(region, level)
                if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                    feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(alphabet) + region * len(alphabet) + char_index
                    phocs[feat_vec_index] = 1
    #Add bigrams
    if phoc_bigrams is not None:
        ngram_features = np.zeros(len(phoc_bigrams)*np.sum(bigram_levels))
        ngram_occupancy = lambda k, n: [float(k) / n, float(k+2) / n]
        for i in range(n-1):
            ngram = word[i:i+2]
            if phoc_bigrams.get(ngram, 0) == 0:
                continue
            occ = ngram_occupancy(i, n)
            for level in bigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    overlap_size = size(overlap(occ, region_occ)) / size(occ)
                    if overlap_size >= 0.5:
                        ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
        phocs[-ngram_features.shape[0]:] = ngram_features        
    return phocs
