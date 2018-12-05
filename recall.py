#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:44:27 2017

@author: tomas
"""

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import json

import os
import easydict
import torch
from misc.dataloader import DataLoader
import misc.datasets as datasets
from train_opts import parse_args
import misc.h5_dataset as h5_dataset
from recall_utils import recall
from misc.utils import average_dictionary

opt = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
opt.augment = 0
opt.quiet = 1
dtp_only = opt.dtp_only

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.device(opt.gpu)

if dtp_only:
    import ctrlfnet_model_dtp as ctrlf
else:
    import ctrlfnet_model as ctrlf

model = ctrlf.CtrlFNet(opt)
model.load_weights(opt.weights)
model.cuda()

args = easydict.EasyDict()
args.nms_overlap = opt.query_nms_overlap
args.score_threshold = opt.score_threshold
args.num_queries = -1
args.score_nms_overlap = opt.score_nms_overlap
args.gpu = True
args.use_external_proposals = int(opt.external_proposals)
args.max_proposals = opt.max_proposals
args.overlap_thresholds = [0.25, 0.5]
args.rpn_nms_thresh = opt.test_rpn_nms_thresh
args.numpy = False
args.num_workers = 6
args.dtp_only = opt.dtp_only

if opt.hyperparam_opt:
    if opt.dtp_only:
        from evaluate_dtp import hyperparam_search
    else:
        from evaluate_torch import hyperparam_search

    print 'performing hyper param search'
    if opt.h5:
        valset = h5_dataset.H5Dataset(opt, split=1)
        opt.num_workers = 0
    else:
        valset = datasets.Dataset(opt, 'val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    hyperparam_search(model, valloader, args, opt, 'all')
    print 'hyper param search done' 
    args.use_external_proposals = int(opt.external_proposals)

r_keys = []
if dtp_only:
    for i in range(1, 4):
        for ot in args.overlap_thresholds:
            r_keys.append('%d_total_recall_%d' % (i, ot * 100))
            
    r_keys += ['%d_total_proposals' % i for i in range(1, 4)]
    
else:
    for i in range(1, 4):
        for ot in args.overlap_thresholds:
            r_keys += ['%d_dtp_recall_%d' %  (i, ot * 100), 
                     '%d_rpn_recall_%d' %  (i, ot * 100), 
                     '%d_total_recall_%d' % (i, ot * 100)]
            
        r_keys += ['%d_total_proposals' % i, '%d_dtp_proposals' % i, '%d_rpn_proposals' % i]

if opt.folds:
    avg = []
    for fold in range(1,5):
        s = opt.weights.find('fold')
        e = s + 5
        opt.weights = opt.weights[:s] + ('fold%d' % fold) + opt.weights[e:]
        opt.fold = fold
        
        if opt.h5:
            testset = h5_dataset.H5Dataset(opt, split=2)
            opt.num_workers = 0
        else:
            testset = datasets.Dataset(opt, 'test')
        
        loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        log, avg = recall(model, loader, args, 0)
        avg.append(avg)
        
    avg = average_dictionary(avg, r_keys)

else:         
    if opt.h5:
        testset = h5_dataset.H5Dataset(opt, split=2)
        opt.num_workers = 0
    else:
        testset = datasets.Dataset(opt, 'test')
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    log, avg = recall(model, loader, args, 0)

log = '\n\n'
log += 'Averaged over folds\n'
log += '--------------------------------\n'
if dtp_only or not args.use_external_proposals:
    for ot in args.overlap_thresholds:
        for i, rtype in enumerate(['a', 't', 'n']):
            total_recall = avg['%d_total_recall_%d' % (i+1, ot * 100)]
            total_np = avg['%d_total_proposals' % (i+1)] / 100
            pargs = (rtype, total_recall, total_np, ot*100)
            rs2 = '%s: total: %.2f, %.2f using %d%% overlap' % pargs
            log += '[%s set] %s\n' % ('test', rs2)
else:
    for ot in args.overlap_thresholds:
        for i, rtype in enumerate(['a', 't', 'n']):
            rpn_recall = avg['%d_rpn_recall_%d' % (i+1, ot * 100)]
            dtp_recall = avg['%d_dtp_recall_%d' % (i+1, ot * 100)]
            total_recall = avg['%d_total_recall_%d' % (i+1, ot * 100)]
    #        pargs = (rtype, rpn_recall, dtp_recall, total_recall, ot*100)
    #        rs2 = '%s: rpn_recall: %.2f, dtp_recall: %.2f, total recall: %.2f using %d%% overlap' % pargs
            rpn_np = avg['%d_rpn_proposals' % (i+1)] / 100
            total_np = avg['%d_total_proposals' % (i+1)] / 100
            dtp_np = avg['%d_dtp_proposals' % (i+1)] / 100
            pargs = (rtype, rpn_recall, rpn_np, dtp_recall, dtp_np, total_recall, total_np, ot*100)
            rs2 = '%s: rpn: %.2f, %d, dtp: %.2f, %d, total: %.2f, %d using %d%% overlap' % pargs
            log += '[%s set] %s\n' % ('test', rs2)
log += '--------------------------------\n'
print log

if opt.save:
    if dtp_only:
        outdir = 'results_recall_dtp/' + opt.weights.split('/')[-1]
    else:
        outdir = 'results_recall/' + opt.weights.split('/')[-1]
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    data = {'avg':avg, 'latex':s}
    with open(outdir + '_data.json', 'w') as fp:
        json.dump(data, fp)
        
    with open(outdir + '_data.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
