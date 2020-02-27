#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:02:35 2019

@author: tomas
"""
import torch
from evaluate_dtp import pairwise_cosine_distances
from evaluate import recall_torch
import misc.box_utils as box_utils
from misc.boxIoU import bbox_overlaps
import numpy as np
from misc.utils import average_dictionary, copy_log

def postprocessing(features, loader, args, model):
    score_nms_overlap = args.score_nms_overlap #For wordness scores
    score_threshold = args.score_threshold
    
    qbs_queries, qbs_qtargets = loader.dataset.get_queries(tensorize=True)
    qbe_queries, gt_targets = [], []
    for li, data in enumerate(loader):
        qbe_queries.append(features[li][4].numpy())
        gt_targets.append(torch.squeeze(data[5]).numpy())

    qbe_queries, qbe_qtargets, _ = loader.dataset.dataset_query_filter(qbe_queries, gt_targets, gt_targets, tensorize=True)
    gt_targets = torch.from_numpy(np.concatenate(gt_targets, axis=0))
    
    all_proposals, log, joint_boxes, db, all_gt_boxes= [], [], [], [], []
    prop2img, recalls = [], []
    db_targets, max_overlaps, amax_overlaps = [], [], []
    n_gt = 0
    overlaps = []
    offset = [0, 0]
    li = 0
    for data in loader:
        roi_scores, eproposal_scores, proposals, embeddings, gt_embed, eproposal_embed = features[li]
        (img, oshape, gt_boxes, external_proposals, gt_embeddings, gt_labels) = data
        
        #boxes are xcycwh from dataloader, convert to x1y1x2y2
        external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())#.round()#.int()
        gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())#.round()#.int()
        img = torch.squeeze(img)
        gt_boxes = torch.squeeze(gt_boxes)
        all_gt_boxes.append(gt_boxes)
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
                
        #Scale boxes back to original size
        sh, sw = img.shape
        img_shape = np.array([oshape[0], oshape[1]])
        scale = float(max(img_shape)) / max(sh, sw)
        scale *= 2
        proposals = torch.round((proposals -1) * scale + 1).int()
        gt_boxes = torch.round((gt_boxes -1) * scale + 1).int()
        
        #calculate the different recalls before NMS
        entry = {}
        
        #Since slicing empty array doesn't work in torch, we need to do this explicitly
        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        proposals = proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))

        dets = torch.cat([proposals.float(), scores.view(-1, 1)], 1)   
        if dets.size(0) <= 1:
            continue
        
        pick = box_utils.nms(dets, score_nms_overlap)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        proposals = proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        
        r1 = recall_torch(proposals.float(), gt_boxes.float(), 0.25)
        r2 = recall_torch(proposals.float(), gt_boxes.float(), 0.5)
        recalls.append((r1, r2))
                    
        for kk in range(len(proposals)):
            prop2img.append(li)

        all_proposals.append(proposals.cpu())
        
        overlap = bbox_overlaps(proposals, gt_boxes)
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
        proposals[:, 0] += offset[1]
        proposals[:, 1] += offset[0]
        proposals[:, 2] += offset[1]
        proposals[:, 3] += offset[0]
        joint_boxes.append(proposals)
        offset[0] += img.shape[0]
        offset[1] += img.shape[1]
        
        db_targets.append(proposal_labels)
        db.append(embeddings)
        log.append(entry)
        li += 1
        
    db = torch.cat(db, dim=0)
    joint_boxes = torch.cat(joint_boxes, dim=0)
#    joint_boxes = torch.FloatTensor(joint_boxes).cuda()
    db_targets = torch.cat(db_targets, dim=0)
    all_gt_boxes = torch.cat(all_gt_boxes, dim=0)
    max_overlaps = torch.cat(max_overlaps, dim=0)
    amax_overlaps = torch.cat(amax_overlaps, dim=0)

    
    assert qbe_queries.shape[0] == qbe_qtargets.shape[0]
    assert qbs_queries.shape[0] == qbs_qtargets.shape[0]
    
    qbe_queries = qbe_queries.cuda()
    qbs_queries = qbs_queries.cuda()
    qbe_dists = pairwise_cosine_distances(qbe_queries, db)
    qbs_dists = pairwise_cosine_distances(qbs_queries, db)

    if args.mAP_gpu:
        gt_targets = gt_targets.cuda()
        qbs_qtargets = qbs_qtargets.cuda()
        qbe_qtargets = qbe_qtargets.cuda()

    else:
        qbe_dists = qbe_dists.cpu()
        qbs_dists = qbs_dists.cpu()
        joint_boxes = joint_boxes.cpu()
        max_overlaps = max_overlaps.cpu()
        amax_overlaps = amax_overlaps.cpu()
        db_targets = db_targets.cpu()
        
    if args.mAP_numpy:
        gt_targets = gt_targets.numpy()
        qbs_qtargets = qbs_qtargets.numpy()
        qbe_qtargets = qbe_qtargets.numpy()        
        qbe_dists = qbe_dists.numpy()
        qbs_dists = qbs_dists.numpy()
        joint_boxes = joint_boxes.numpy()
        max_overlaps = max_overlaps.numpy()
        amax_overlaps = amax_overlaps.numpy()
        db_targets = db_targets.numpy()
        
    prop2img = np.array(prop2img)
    recalls = np.array(recalls)
        
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
                    
    return (qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets, gt_targets, prop2img,
            joint_boxes, all_proposals, recalls, max_overlaps, amax_overlaps)


def postprocessing_dtp(features, loader, args, model):
    score_nms_overlap = args.score_nms_overlap #For wordness scores
    score_threshold = args.score_threshold
    overlap_thresholds = args.overlap_thresholds
    num_queries = args.num_queries
    
    all_gt_boxes = []
    joint_boxes = []
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
    prop2img, recalls = [], []
    all_proposals = []
    overlaps = []
    all_gt_boxes = []
    db_targets = []
    db = []
    joint_boxes = []
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
        
        #Scale boxes back to original size
        sh, sw = img.shape
        img_shape = np.array([oshape[0], oshape[1]])
        scale = float(max(img_shape)) / max(sh, sw)
        scale *= 2
        dtp_proposals = torch.round((dtp_proposals -1) * scale + 1).int()
        gt_boxes = torch.round((gt_boxes -1) * scale + 1).int()
        
        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        dtp_proposals = dtp_proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))
        
        dets = torch.cat([dtp_proposals.float(), scores], 1)   
        pick = box_utils.nms(dets, score_nms_overlap, args.nms_max_boxes)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        dtp_proposals = dtp_proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        
        r1 = recall_torch(dtp_proposals.float(), gt_boxes.float(), 0.25)
        r2 = recall_torch(dtp_proposals.float(), gt_boxes.float(), 0.5)
        recalls.append((r1, r2))
        
        for kk in range(len(dtp_proposals)):
            prop2img.append(li)

        all_proposals.append(dtp_proposals.cpu())

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
#    print qbe_queries.shape, db.shape, qbs_queries.shape
    qbe_dists = pairwise_cosine_distances(qbe_queries, db)
    qbs_dists = pairwise_cosine_distances(qbs_queries, db)

    if args.mAP_gpu:
        gt_targets = gt_targets.cuda()
        qbs_qtargets = qbs_qtargets.cuda()
        qbe_qtargets = qbe_qtargets.cuda()

    else:
        qbe_dists = qbe_dists.cpu()
        qbs_dists = qbs_dists.cpu()
        db_targets = db_targets.cpu()
        joint_boxes = joint_boxes.cpu()
        max_overlaps = max_overlaps.cpu()
        amax_overlaps = amax_overlaps.cpu()
        
    if args.mAP_numpy:
        gt_targets = gt_targets.numpy()
        qbs_qtargets = qbs_qtargets.numpy()
        qbe_qtargets = qbe_qtargets.numpy()        
        qbe_dists = qbe_dists.numpy()
        qbs_dists = qbs_dists.numpy()
        db_targets = db_targets.numpy()
        joint_boxes = joint_boxes.numpy()
        max_overlaps = max_overlaps.numpy()
        amax_overlaps = amax_overlaps.numpy()
        
    prop2img = np.array(prop2img)
    recalls = np.array(recalls)
        
    return (qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets, gt_targets, prop2img,
            joint_boxes, all_proposals, recalls, max_overlaps, amax_overlaps)
