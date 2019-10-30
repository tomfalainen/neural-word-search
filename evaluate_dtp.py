#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:59:41 2018

@author: tomas
"""
import torch 
import numpy as np
np.errstate(divide='ignore', invalid='ignore')
import misc.box_utils as box_utils
from misc.boxIoU import bbox_overlaps
from evaluate import extract_features, calcuate_mAPs, recalls, pairwise_cosine_distances

def hyperparam_search(model, valloader, args, opt, score_vars='all'):
    variables = ['score_nms_overlap', 'score_threshold']
    ranges = [np.arange(0.1, 0.71, 0.1), np.arange(0.0, 0.1, 0.01)]
    best = {}
    for (variable, vals) in zip(variables, ranges):
        maps = []
        best_score = 0.0
        best_val = -1
        for v in vals:
            args[variable] = v
            log, rf, rt = mAP(model, valloader, args, 0)

            if score_vars == '50':
                score = (rt.mAP_qbe_50 + rt.mAP_qbs_50) / 2    
            else:
                score = (rt.mAP_qbe_50 + rt.mAP_qbs_50 + rt.mAP_qbe_25 + rt.mAP_qbs_25) / 4
                
            maps.append([rt.mAP_qbe_50, rt.mAP_qbs_50])
            if score > best_score:
                best_score = score
                best_val = v
    
        best[variable] = (best_score, best_val)
        args[variable] = opt[variable]
        
    for v in variables:
        args[v] = best[v][1]
    
def mAP(model, loader, args, it):
    features = extract_features(model, loader, args, args.numpy)
    recall = 3   
    split = loader.dataset.split
    
    if loader.dataset.dataset == 'iam':
        args.overlap_thresholds = [0.25, 0.5]
        res = mAP_eval(features, loader, args, model)
        total_recall = np.mean([e['%d_total_recall_25' % recall] for e in res.log])
        pargs = (res.mAP_qbe_25*100, res.mAP_qbs_25*100, total_recall*100)
        rs1 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, With DTP 25%% overlap' % pargs
        total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res.log])
        pargs = (res.mAP_qbe_50*100, res.mAP_qbs_50*100, total_recall*100)
        rs2 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, With DTP 50%% overlap' % pargs
        log = '--------------------------------\n'
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs1)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs2)
        log += '--------------------------------\n'
        return log, res, res
        
    else:
        args.overlap_thresholds = [0.25, 0.5]
        res_true = mAP_eval(features, loader, args, model)
        total_recall = np.mean([e['%d_total_recall_25' % recall] for e in res_true.log])
        pargs = (res_true.mAP_qbe_25*100, res_true.mAP_qbs_25*100, total_recall*100)
        rs3 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, 25%% overlap' % pargs
        total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res_true.log])
        pargs = (res_true.mAP_qbe_50*100, res_true.mAP_qbs_50*100, total_recall*100)
        rs4 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, 50%% overlap' % pargs
        log = '--------------------------------\n'
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs3)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs4)
        log += '--------------------------------\n'
        return log, res_true, res_true
    
def mAP_eval(features, loader, args, model):
    d = postprocessing(features, loader, args, model)
    d += (args, )
    results = calcuate_mAPs(*d)
    return results

def postprocessing(features, loader, args, model):
    score_nms_overlap = args.score_nms_overlap #For wordness scores
    score_threshold = args.score_threshold
    overlap_thresholds = args.overlap_thresholds
    num_queries = args.num_queries
    
    all_gt_boxes = []
    joint_boxes = []
    log = []
    qbs_queries, qbs_qtargets = loader.dataset.get_queries(tensorize=True)
    
    qbe_queries, gt_targets = [], []
    for li, data in enumerate(loader):
        qbe_queries.append(features[li][1].numpy())
        gt_targets.append(torch.squeeze(data[5]).numpy())

    qbe_queries, qbe_qtargets, _ = loader.dataset.dataset_query_filter(qbe_queries, gt_targets, gt_targets, tensorize=True)
    gt_targets = torch.from_numpy(np.concatenate(gt_targets, axis=0))
    
    if num_queries < 1:
        num_queries = len(qbe_queries) + len(qbs_queries) + 1

    qbe_queries = qbe_queries[:num_queries]
    qbs_queries = qbs_queries[:num_queries]
    qbe_qtargets = qbe_qtargets[:num_queries]
    qbs_qtargets = qbs_qtargets[:num_queries]
    
    max_overlaps, amax_overlaps = [], []
    overlaps = []
    all_gt_boxes = []
    db_targets = []
    db = []
    joint_boxes = []
    log = []
    offset = [0, 0]
    n_gt = 0
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
        recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '1_total')
        
        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        dtp_proposals = dtp_proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))
        recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '2_total')
        
        dets = torch.cat([dtp_proposals.float(), scores], 1)   
        pick = box_utils.nms(dets, score_nms_overlap)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        dtp_proposals = dtp_proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_total')
        
        overlap = bbox_overlaps(dtp_proposals, gt_boxes)
        overlaps.append(overlap)
        max_gt_overlap, amax_gt_overlap = overlap.max(dim=1)
        proposal_labels = torch.Tensor([gt_labels[i] for i in amax_gt_overlap])
        proposal_labels = proposal_labels.cuda()    
        mask = overlap.sum(dim=1) == 0
        proposal_labels[mask] = loader.dataset.get_vocab_size() + 1    
        
        max_overlaps.append(max_gt_overlap)
        amax_overlaps.append(amax_gt_overlap + n_gt)
        n_gt += len(gt_boxes)
        
        # Artificially make a huge image containing all the boxes to be able to
        # perform nms on distance to query
        dtp_proposals[:, 0] += offset[1]
        dtp_proposals[:, 1] += offset[0]
        dtp_proposals[:, 2] += offset[1]
        dtp_proposals[:, 3] += offset[0]
        joint_boxes.append(dtp_proposals)
        
        offset[0] += img.shape[0]
        offset[1] += img.shape[1]
        
        db_targets.append(proposal_labels)
        db.append(embeddings)
        log.append(entry)
        
    db = torch.cat(db, dim=0)
    db_targets = torch.cat(db_targets, dim=0)
    joint_boxes = torch.cat(joint_boxes, dim=0)
    max_overlaps = torch.cat(max_overlaps, dim=0)
    amax_overlaps = torch.cat(amax_overlaps, dim=0)
    all_gt_boxes = torch.cat(all_gt_boxes, dim=0)
    
    assert qbe_queries.shape[0] == qbe_qtargets.shape[0]
    assert qbs_queries.shape[0] == qbs_qtargets.shape[0]
    assert db.shape[0] == db_targets.shape[0]
    
    qbe_queries = qbe_queries.cuda()
    qbs_queries = qbs_queries.cuda()
    qbe_dists = pairwise_cosine_distances(qbe_queries, db)
    qbs_dists = pairwise_cosine_distances(qbs_queries, db)
    
    qbe_dists = qbe_dists.cpu()
    qbs_dists = qbs_dists.cpu()
    db_targets = db_targets.cpu()
    joint_boxes = joint_boxes.cpu()
    max_overlaps = max_overlaps.cpu()
    amax_overlaps = amax_overlaps.cpu()
        
    gt_targets = gt_targets.numpy()
    qbs_qtargets = qbs_qtargets.numpy()
    qbe_qtargets = qbe_qtargets.numpy()        
    qbe_dists = qbe_dists.numpy()
    qbs_dists = qbs_dists.numpy()
    db_targets = db_targets.numpy()
    joint_boxes = joint_boxes.numpy()
    max_overlaps = max_overlaps.numpy()
    amax_overlaps = amax_overlaps.numpy()
        
    return (qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets, gt_targets, 
            joint_boxes, max_overlaps, amax_overlaps, log)

