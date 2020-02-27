#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:28:54 2016

@author: tomas
"""


from xml.dom.minidom import getDOMImplementation
import subprocess
import os
import easydict
import numpy as np
import torch

import misc.h5_dataset as h5_dataset        
import misc.dataloader as dataloader
import misc.datasets as datasets
from train_opts import parse_args

import misc.box_utils as box_utils
import misc.dataset_loader as dl

def write_to_xml(data, query_texts, qbs_dists, joint_boxes, prop2img,
                     all_proposals, nms_overlap, dataset, cutoff, mode='qbs'):
    impl = getDOMImplementation()
    newdoc = impl.createDocument(None, "wordLocations", None)
    top_element = newdoc.documentElement
    top_element.setAttribute('dataset', dataset.capitalize())

    for dists, qtext in zip(qbs_dists, query_texts):
        sim = 1 - dists
        dets = np.hstack((joint_boxes, sim[:, np.newaxis]))
        pick = box_utils.nms_np(dets, nms_overlap)
        dists = dists[pick]
        I = np.argsort(dists)
        p = prop2img[pick][I]
        ap = all_proposals[pick][I]
        
        if cutoff > 0:
            ap = ap[:cutoff]
            p = p[:cutoff]
        
        for box, img_ind in zip(ap, p):
            datum = data[img_ind]
            spot = newdoc.createElement('spot')
            wfile = datum['id']#.split('/')[-1]
            x, y = box[0].item(), box[1].item()
            w = box[2].item() - x
            h = box[3].item() - y
            if mode == 'qbs':
                spot.setAttribute('word', qtext.encode('utf-8').upper())
            else:
                spot.setAttribute('word', qtext)
            spot.setAttribute('image', wfile)
            spot.setAttribute('x', str(x))
            spot.setAttribute('y', str(y))
            spot.setAttribute('w', str(w))
            spot.setAttribute('h', str(h))
            top_element.appendChild(spot)
        
    with open("botany_konz_eval/data/%s_results_%s.xml" % (dataset, mode), 'wb') as f:
        newdoc.writexml(f, addindent='  ',newl='\n', encoding='utf-8')
        

#%%
opt = parse_args()
opt.augment = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

if opt.h5:
    testset = h5_dataset.H5Dataset(opt, split=2)
    valset = h5_dataset.H5Dataset(opt, split=1)
    opt.num_workers = 0
else:
    testset = datasets.Dataset(opt, 'test')
    valset = datasets.Dataset(opt, 'val')
    
loader = dataloader.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
valloader = dataloader.DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.device(opt.gpu)

opt.vocab_size = testset.get_vocab_size()
opt.embedding_dim = testset.embedding_dim

if opt.dtp_only:
    import ctrlfnet_model_dtp as ctrlf
    from evaluate_dtp import hyperparam_search
    from evaluate_dtp import extract_features
    from botany_konz_eval.hkws16_utils import postprocessing_dtp as postprocessing

else:
    import ctrlfnet_model as ctrlf
    from evaluate import hyperparam_search
    from evaluate import extract_features
    from botany_konz_eval.hkws16_utils import postprocessing

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
args.mAP_gpu = False
args.mAP_numpy = True
args.nms_max_boxes = opt.nms_max_boxes
args.dtp_only = opt.dtp_only
args.test_time_augmentation = 0
args.qbs_only = 1

if opt.hyperparam_opt:
    args.qbs_only = 0
    hyperparam_search(model, valloader, args, opt, score_vars='all')
    args.qbs_only = 1

args.nms_max_boxes = None

features = extract_features(model, loader, args, args.numpy)
d = postprocessing(features, loader, args, model)
(qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets, gt_targets, prop2img,
            joint_boxes, all_proposals, recalls, max_overlaps, amax_overlaps) = d
print recalls.mean(axis=0)

query_texts = testset.qtexts
nms_overlap = args.nms_overlap
data = testset.data_split
all_proposals = torch.cat(all_proposals)

cutoff = 1000
write_to_xml(data, query_texts, qbs_dists, joint_boxes, prop2img, all_proposals, nms_overlap, opt.dataset, cutoff)

log = '[test set QbS]'
for ot in args.overlap_thresholds:
    sub = "botany_konz_eval/data/%s_results_qbs.xml" % opt.dataset
    gt = 'data/%s/%s_Test_GT_SegFree_QbS.xml' % (opt.dataset, opt.dataset.capitalize())
    cmd = 'botany_konz_eval/icfhr16kws_evaluation_toolkit/EvalICFHR16KWS.sh -thr %.2f %s %s' % (ot, gt, sub)
    s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    k = s.find('ALL')
    mAP = float(s[k:].split()[-1])
    log += ', %s %d%%' % (mAP*100, ot*100)
log += '\n'    

mask = []
query_files = []
for d in data:
    for r in d['regions']:
        if r.has_key('query_file'):    
            mask.append(1)
            query_files.append(r['query_file'].split('/')[-1])
        else:
            mask.append(0)

mask = np.array(mask, dtype=np.bool)
qbe = qbe_dists[mask]  
write_to_xml(data, query_files, qbe, joint_boxes, prop2img, all_proposals, nms_overlap, opt.dataset, cutoff, 'qbe')

log += '[test set QbE]'
for ot in args.overlap_thresholds:
    sub = "botany_konz_eval/data/%s_results_qbe.xml" % opt.dataset
    gt = 'data/%s/%s_Test_GT_SegFree_QbE.xml' % (opt.dataset, opt.dataset.capitalize())
    cmd = 'botany_konz_eval/icfhr16kws_evaluation_toolkit/EvalICFHR16KWS.sh -thr %.2f %s %s' % (ot, gt, sub)
    s = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    k = s.find('ALL')
    mAP = float(s[k:].split()[-1])
    log += ', %s %d%%' % (mAP*100, ot*100)
log += '\n'    
print log      

