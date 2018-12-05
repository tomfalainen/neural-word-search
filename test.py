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
import ctrlfnet_model as ctrlf
from train_opts import parse_args
import misc.h5_dataset as h5_dataset
from evaluate import hyperparam_search
from misc.utils import average_dictionary, copy_log

opt = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
opt.augment = 0
dtp_only = opt.dtp_only
if dtp_only:
    from evaluate_dtp import mAP
else:
    from evaluate import mAP

if opt.h5:
    testset = h5_dataset.H5Dataset(opt, split=2)
    opt.num_workers = 0
else:
    testset = datasets.Dataset(opt, 'test')

loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.device(opt.gpu)

# initialize the Ctrl-F-Net model object
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

r_keys = ['3_dtp_recall_50', '3_rpn_recall_50', '3_total_recall_50', 
          '3_dtp_recall_25', '3_rpn_recall_25', '3_total_recall_25']

keys = []
for ot in args.overlap_thresholds:
    keys += ['mAP_qbe_%d' % (ot * 100), 'mAP_qbs_%d' % (ot * 100),
             'mR_qbe_%d' % (ot*100), 'mR_qbs_%d' % (ot*100)]
    
if opt.hyperparam_opt:
    print 'performing hyper param search'
    if opt.h5:
        valset = h5_dataset.H5Dataset(opt, split=1)
        opt.num_workers = 0
    else:
        valset = datasets.Dataset(opt, 'val')
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    hyperparam_search(model, valloader, args, opt, 'all')
    print 'hyper param search done' 
    
if opt.folds:
    rts, rfs = [], []
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

        model = ctrlf.CtrlFNet(opt)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        log, rf, rt = mAP(model, loader, args, 0)
        print(log)
        rt['log'] = average_dictionary(rt['log'], r_keys)
        rf['log'] = average_dictionary(rf['log'], ['3_total_recall_50', '3_total_recall_25'])
        copy_log(rt, rf)
        rts.append(rt)
        rfs.append(rf)

    avg_rts = average_dictionary(rts, keys + r_keys)
    avg_rfs = average_dictionary(rfs, keys + ['3_total_recall_50', '3_total_recall_25'])

else:     
    log, rf, rt = mAP(model, loader, args, 0)
    print(log)
    #average pagewise recalls
    rt['log'] = average_dictionary(rt['log'], r_keys)
    copy_log(rt)
    avg_rts = rt

def final_log(dicts, keys, title):
    res = average_dictionary(dicts, keys, False, True)
    pargs = (res.mAP_qbe_25, res.mAP_qbs_25, res.mR_qbe_25, res.mR_qbs_25)
    s1 = 'QbE mAP: %.1f, QbS mAP: %.1f, QbE mR: %.1f, QbS mR: %.1f, 25%% overlap' % pargs
    pargs = (res.mAP_qbe_50, res.mAP_qbs_50, res.mR_qbe_50, res.mR_qbs_50)
    s2 = 'QbE mAP: %.1f, QbS mAP: %.1f, QbE mR: %.1f, QbS mR: %.1f, 50%% overlap' % pargs
    log = '%s\n--------------------------------\n' % title
    log += '[test set] %s\n' % s1
    log += '[test set] %s\n' % s2
    log += '--------------------------------\n'
    return log

if opt.folds:
    print opt.weights
    print final_log(avg_rts, keys + r_keys, 'With DTP')
    
    if not dtp_only and opt.dataset != 'iam':
        print final_log(avg_rfs, keys, 'RPN only')
    
if opt.save:
    if dtp_only:
        outdir = 'results_dtp/' + opt.weights.split('/')[-1]
    elif opt.ghosh:
        outdir = 'results_ghosh/' + opt.weights.split('/')[-1]
    else:
        outdir = 'results/' + opt.weights.split('/')[-1]
        
    data = {'avg_rts':avg_rts, 'avg_rfs': avg_rfs}
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    with open(outdir + '_data.json', 'w') as fp:
        json.dump(data, fp)
        
    with open(outdir + '_data.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
