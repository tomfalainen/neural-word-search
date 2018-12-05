#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:30:11 2018

@author: tomas
"""

import torch
import numpy as np
import evaluate_torch as et
import evaluate_dtp as edtp
import misc.box_utils as box_utils
from misc.utils import average_dictionary

def recall(model, loader, args, it):
    split = loader.dataset.split
    keys = []
    if args.dtp_only:
        features = edtp.extract_features(model, loader, args, args.numpy)
        res = postprocessing_dtp(features, loader, args)
        for ot in args.overlap_thresholds:
            keys += ['%d_total_recall_%d' % (i, ot*100) for i in range(1, 4)]
            
        keys += ['%d_total_proposals' % i for i in range(1, 4)]
    else:
        features = et.extract_features(model, loader, args, args.numpy)
        res = postprocessing(features, loader, args)
        for i in range(1, 4):
            for ot in args.overlap_thresholds:
                keys += ['%d_dtp_recall_%d' %  (i, ot * 100), 
                         '%d_rpn_recall_%d' %  (i, ot * 100), 
                         '%d_total_recall_%d' % (i, ot * 100)]
                
        for i in range(1, 4):
            keys += ['%d_total_proposals' % i, '%d_dtp_proposals' % i, '%d_rpn_proposals' % i]
    
    avg = average_dictionary(res, keys, scale=True)
    log = '\n\n Fold %d\n' % loader.dataset.fold
    log += '--------------------------------\n'
    if args.dtp_only:
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
                rpn_np = avg['%d_rpn_proposals' % (i+1)] / 100
                total_np = avg['%d_total_proposals' % (i+1)] / 100
                dtp_np = avg['%d_dtp_proposals' % (i+1)] / 100
    #            pargs = (rtype, rpn_recall, dtp_recall, total_recall, ot*100)
                pargs = (rtype, rpn_recall, rpn_np, dtp_recall, dtp_np, total_recall, total_np, ot*100)
    #            rs2 = '%s: rpn_recall: %.2f, dtp_recall: %.2f, total recall: %.2f using %d%% overlap' % pargs
                rs2 = '%s: rpn: %.2f, %d, dtp: %.2f, %d, total: %.2f, %d using %d%% overlap' % pargs
                log += '[%s set] %s\n' % (split, rs2)
    log += '--------------------------------\n'
    return log, avg
        
def postprocessing(features, loader, args):
    score_nms_overlap = args.score_nms_overlap #For wordness scores
    score_threshold = args.score_threshold
    overlap_thresholds = args.overlap_thresholds
    log = []
    gt_targets = []
    for li, data in enumerate(loader):
        gt_targets.append(torch.squeeze(data[5]).numpy())

    gt_targets = torch.from_numpy(np.concatenate(gt_targets, axis=0))
    for li, data in enumerate(loader):
        roi_scores, eproposal_scores, proposals, embeddings, gt_embed, eproposal_embed = features[li]
        (img, oshape, gt_boxes, external_proposals, gt_embeddings, gt_labels) = data
        
        #boxes are xcycwh from dataloader, convert to x1y1x2y2
        external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
        gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())
        img = torch.squeeze(img)
        gt_boxes = torch.squeeze(gt_boxes)
        gt_labels = torch.squeeze(gt_labels)
        gt_embeddings = torch.squeeze(gt_embeddings)
        
        gt_boxes = gt_boxes.cuda()
        gt_embeddings = gt_embeddings.cuda()
        gt_labels = gt_labels.cuda()
        roi_scores = roi_scores.cuda()
        eproposal_scores = eproposal_scores.cuda()
        eproposal_embed = eproposal_embed.cuda()
        proposals = proposals.cuda()
        embeddings = embeddings.cuda()
        gt_embed = gt_embed.cuda()
        external_proposals = external_proposals.cuda()
        
        #convert to probabilities with sigmoid
        scores = 1 / (1 + torch.exp(-roi_scores))
        
        if args.use_external_proposals:
            eproposal_scores = 1 / (1 + torch.exp(-eproposal_scores))
            scores = torch.cat((scores, eproposal_scores), 0)
            proposals = torch.cat((proposals, external_proposals), 0) 
            embeddings = torch.cat((embeddings, eproposal_embed), 0)
                
        #calculate the different recalls before NMS
        entry = {}
        et.recalls(proposals, gt_boxes, overlap_thresholds, entry, '1_total')
        entry['1_total_proposals'] = proposals.size(0)
        
        #Since slicing empty array doesn't work in torch, we need to do this explicitly
        if args.use_external_proposals:
            nrpn = len(roi_scores)
            rpn_proposals = proposals[:nrpn]
            dtp_proposals = proposals[nrpn:]
            et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '1_dtp')
            et.recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '1_rpn')
            entry['1_dtp_proposals'] = dtp_proposals.size(0)
            entry['1_rpn_proposals'] = rpn_proposals.size(0)
        
        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        proposals = proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))
        et.recalls(proposals, gt_boxes, overlap_thresholds, entry, '2_total')
        entry['2_total_proposals'] = proposals.size(0)

        if args.use_external_proposals:
            rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
            dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '2_dtp')
            et.recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '2_rpn')
            entry['2_dtp_proposals'] = dtp_proposals.size(0)
            entry['2_rpn_proposals'] = rpn_proposals.size(0)

        dets = torch.cat([proposals.float(), scores], 1)   
        if dets.size(0) <= 1:
            continue
        
        pick = box_utils.nms(dets, score_nms_overlap)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        proposals = proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        et.recalls(proposals, gt_boxes, overlap_thresholds, entry, '3_total')
        entry['3_total_proposals'] = proposals.size(0)
        if args.use_external_proposals:
            nrpn = rpn_proposals.size(0)
            tmp = tt.view(-1, 1).expand(tt.size(0), 4)
            rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
            dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_dtp')
            et.recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '3_rpn')
            entry['3_dtp_proposals'] = dtp_proposals.size(0)
            entry['3_rpn_proposals'] = rpn_proposals.size(0)
        
        log.append(entry)
    
    #A hack for some printing compatability
    if not args.use_external_proposals:
        keys = []
        for i in range(1, 4):
            for ot in args.overlap_thresholds:
                keys += ['%d_dtp_recall_%d' %  (i, ot * 100), 
                         '%d_rpn_recall_%d' %  (i, ot * 100)]
                
        for entry in log:
            for key in keys:
                if not entry.has_key(key):
                    entry[key] = entry['1_total_recall_50']
                 
        keys = []
        for i in range(1, 4):
            keys += ['%d_dtp_proposals' % i, '%d_rpn_proposals' % i]
        
        for entry in log:
            for key in keys:
                if not entry.has_key(key):
                    entry[key] = entry['1_total_proposals']
                
        
    return log

def postprocessing_dtp(features, loader, args):
    score_nms_overlap = args.score_nms_overlap #For wordness scores
    score_threshold = args.score_threshold
    overlap_thresholds = args.overlap_thresholds
    
    all_gt_boxes = []
    log = []
    gt_targets = []
    for li, data in enumerate(loader):
        gt_targets.append(torch.squeeze(data[5]).numpy())

    gt_targets = torch.from_numpy(np.concatenate(gt_targets, axis=0))
    log = []
    for li, data in enumerate(loader):
        scores, gt_embed, embeddings = features[li]
        (img, oshape, gt_boxes, dtp_proposals, gt_embeddings, gt_labels) = data
        
        #boxes are xcycwh from dataloader, convert to x1y1x2y2
        dtp_proposals = box_utils.xcycwh_to_x1y1x2y2(dtp_proposals[0].float())#.round()#.int()
        gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())#.round()#.int()
        img = torch.squeeze(img)
        gt_boxes = torch.squeeze(gt_boxes)
        all_gt_boxes.append(gt_boxes)
        gt_labels = torch.squeeze(gt_labels)
        gt_embeddings = torch.squeeze(gt_embeddings)
        
        gt_boxes = gt_boxes.cuda()
        gt_embeddings = gt_embeddings.cuda()
        gt_labels = gt_labels.cuda()
        scores = scores.cuda()
        embeddings = embeddings.cuda()
        gt_embed = gt_embed.cuda()
        dtp_proposals = dtp_proposals.cuda()
        
        #convert to probabilities with sigmoid
        scores = 1 / (1 + torch.exp(-scores))
        
        #calculate the different recalls before NMS
        entry = {}
        et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '1_total')
        entry['1_total_proposals'] = dtp_proposals.size(0)

        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        dtp_proposals = dtp_proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))
        et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '2_total')
        entry['2_total_proposals'] = dtp_proposals.size(0)

        dets = torch.cat([dtp_proposals.float(), scores], 1)   
        pick = box_utils.nms(dets, score_nms_overlap)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        dtp_proposals = dtp_proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        et.recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_total')
        entry['3_total_proposals'] = dtp_proposals.size(0)
        log.append(entry)
        
    return log