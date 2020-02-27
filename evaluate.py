#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:56:02 2017

@author: tomas
"""
from contextlib import closing
from multiprocessing import Pool as PyPool
import easydict
import torch 
import numpy as np
np.errstate(divide='ignore', invalid='ignore')
import misc.box_utils as box_utils
from misc.boxIoU import bbox_overlaps

def pairwise_cosine_distances(x, y, batch_size=1000):
    """
    Input: x is a Nxd matrix
           y is an Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the cosine distance between x[i,:] and y[j,:]
    """
    y_norm = y.norm(2, dim=1)
    def cos(x):
        x_norm = x.norm(2, dim=1)
        return 1 - torch.mm(x, torch.transpose(y, 0, 1)) / torch.ger(x_norm, y_norm)
        
    dist = []
    for v in x.split(batch_size):
        dist.append(cos(v))    
    dist = torch.cat(dist, dim=0)
    return dist

def hyperparam_search(model, valloader, args, opt, score_vars='all'):
    variables = ['score_nms_overlap', 'score_threshold', 'test_rpn_nms_thresh']
    ranges = [np.arange(0.1, 0.71, 0.1), np.arange(0.0, 0.1, 0.01), np.arange(0.1, 0.51, 0.1)]
    best = {}
    use_dtp = args.use_external_proposals
    for (variable, vals) in zip(variables, ranges):
        maps = []
        best_score = 0.0
        best_val = -1
        for v in vals:
            args[variable] = v
            log, rf, rt = mAP(model, valloader, args, 0)
            if not use_dtp:
                rt = rf

            if score_vars == '50':
                score = (rt.mAP_qbe_50 + rt.mAP_qbs_50) / 2    
            else:
                score = (rt.mAP_qbe_50 + rt.mAP_qbs_50 + rt.mAP_qbe_25 + rt.mAP_qbs_25) / 4
                
            print v, score
                
            maps.append([rt.mAP_qbe_50, rt.mAP_qbs_50])
            if score > best_score:
                best_score = score
                best_val = v
    
        best[variable] = (best_score, best_val)
        args[variable] = opt[variable]
        
    for v in variables:
        args[v] = best[v][1]
    
    args.use_external_proposals = use_dtp

def my_unique(tensor1d):
    """ until pytorch adds this """
    t, idx = np.unique(tensor1d.cpu().numpy(), return_inverse=True)
    return t.shape[0]

def recall_torch(proposals, gt_boxes, ot):
    if proposals.nelement() == 0:
        return 0.0
    overlap = bbox_overlaps(proposals, gt_boxes)
    vals, inds = overlap.max(dim=1)
    i = vals >= ot
    covered = my_unique(inds[i])
    recall = float(covered) / float(gt_boxes.size(0))
    return recall

def recalls(proposals, gt_boxes, overlap_thresholds, entry, key):
    for ot in overlap_thresholds:
        entry['%s_recall_%d' % (key, ot*100)] = recall_torch(proposals, gt_boxes, ot)

def extract_features(model, loader, args, numpy=True):
    outputs = []    
    model.eval()
    for data in loader:
        (img, oshape, gt_boxes, external_proposals, gt_embeddings, gt_labels) = data
        
        if args.max_proposals == -1:
            model.setTestArgs({'rpn_nms_thresh':args.rpn_nms_thresh,'max_proposals':external_proposals.size(1)})
        else:
            model.setTestArgs({'rpn_nms_thresh':args.rpn_nms_thresh,'max_proposals':args.max_proposals})
        
        input = (img, gt_boxes[0].float(), external_proposals[0].float())
        out = model.evaluate(input, args.gpu, numpy)
        outputs.append(out)
         
    model.train()
    return outputs

def mAP(model, loader, args, it):
    features = extract_features(model, loader, args, args.numpy)
    recall = 3   
    split = loader.dataset.split
    args.overlap_thresholds = [0.25, 0.5]
    if loader.dataset.dataset == 'iam':
        res = mAP_eval(features, loader, args, model)
        total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res.log])
        rpn_recall = np.mean([e['%d_rpn_recall_50' % recall] for e in res.log])
        pargs = (res.mAP_qbe_50*100, res.mAP_qbs_50*100, total_recall*100, rpn_recall*100)
        rs2 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f, With DTP 50%% overlap' % pargs
        log = '--------------------------------\n'
#        log += '[%s set iter %d] %s\n' % (split, it + 1, rs1)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs2)
        log += '--------------------------------\n'
        return log, res, res
        
    else:
        args.use_external_proposals = False
        res = mAP_eval(features, loader, args, model)
        
        total_recall = np.mean([e['%d_total_recall_25' % recall] for e in res.log])
        rpn_recall = total_recall #np.mean([e['%d_rpn_recall_25' % recall] for e in res.log])
        pargs = (res.mAP_qbe_25*100, res.mAP_qbs_25*100, total_recall*100, rpn_recall*100)
        rs1 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f 25%% overlap' % pargs
        total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res.log])
        rpn_recall = total_recall
        pargs = (res.mAP_qbe_50*100, res.mAP_qbs_50*100, total_recall*100, rpn_recall*100)
        rs2 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f 50%% overlap' % pargs
        
        args.use_external_proposals = True
        res_true = mAP_eval(features, loader, args, model)
        total_recall = np.mean([e['%d_total_recall_25' % recall] for e in res_true.log])
        rpn_recall = np.mean([e['%d_rpn_recall_25' % recall] for e in res_true.log])
        pargs = (res_true.mAP_qbe_25*100, res_true.mAP_qbs_25*100, total_recall*100, rpn_recall*100)
        rs3 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f, With DTP 25%% overlap' % pargs
        total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res_true.log])
        rpn_recall = np.mean([e['%d_rpn_recall_50' % recall] for e in res_true.log])
        pargs = (res_true.mAP_qbe_50*100, res_true.mAP_qbs_50*100, total_recall*100, rpn_recall*100)
        rs4 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f, With DTP 50%% overlap' % pargs
        log = '--------------------------------\n'
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs1)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs2)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs3)
        log += '[%s set iter %d] %s\n' % (split, it + 1, rs4)
        log += '--------------------------------\n'
        return log, res, res_true
    
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
        qbe_queries.append(features[li][4].numpy())
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
        roi_scores, eproposal_scores, proposals, embeddings, gt_embed, eproposal_embed = features[li]
        (img, oshape, gt_boxes, external_proposals, gt_embeddings, gt_labels) = data
        
        #boxes are xcycwh from dataloader, convert to x1y1x2y2
        external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
        gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())
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
        
        #calculate the different recalls before NMS
        entry = {}
        recalls(proposals, gt_boxes, overlap_thresholds, entry, '1_total')
        
        #Since slicing empty array doesn't work in torch, we need to do this explicitly
        if args.use_external_proposals:
            nrpn = len(roi_scores)
            rpn_proposals = proposals[:nrpn]
            dtp_proposals = proposals[nrpn:]
            recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '1_dtp')
            recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '1_rpn')
        
        threshold_pick = torch.squeeze(scores > score_threshold)
        scores = scores[threshold_pick]
        tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
        proposals = proposals[tmp].view(-1, 4)
        embeddings = embeddings[threshold_pick.view(-1, 1).expand(threshold_pick.size(0), embeddings.size(1))].view(-1, embeddings.size(1))
        recalls(proposals, gt_boxes, overlap_thresholds, entry, '2_total')
        
        if args.use_external_proposals:
            rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
            dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '2_dtp')
            recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '2_rpn')

#        print proposals.shape, scores.shape

        dets = torch.cat([proposals.float(), scores.unsqueeze(1)], 1)   
        if dets.size(0) <= 1:
            continue
        
        pick = box_utils.nms(dets, score_nms_overlap, args.nms_max_boxes)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1 

        proposals = proposals[pick]
        embeddings = embeddings[pick]
        scores = scores[pick]
        recalls(proposals, gt_boxes, overlap_thresholds, entry, '3_total')
        if args.use_external_proposals:
            nrpn = rpn_proposals.size(0)
            tmp = tt.view(-1, 1).expand(tt.size(0), 4)
            rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
            dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_dtp')
            recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '3_rpn')
            
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
                    
    return (qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets, gt_targets, 
            joint_boxes, max_overlaps, amax_overlaps, log)

def calcuate_mAPs(qbe_dists, qbe_qtargets, qbs_dists, qbs_qtargets, db_targets,
                  gt_targets, joint_boxes, max_overlaps, amax_overlaps, log, args):
    
    results = easydict.EasyDict()
    results['log'] = log
    
    for ot in args.overlap_thresholds:
        mAP_qbe, mR_qbe = mAP_parallel(qbe_dists, qbe_qtargets, db_targets,
                                          gt_targets, joint_boxes, max_overlaps, 
                                          amax_overlaps, args.nms_overlap, ot, args.num_workers, args)
        mAP_qbs, mR_qbs = mAP_parallel(qbs_dists, qbs_qtargets, db_targets,
                                          gt_targets, joint_boxes, max_overlaps, 
                                          amax_overlaps, args.nms_overlap, ot, args.num_workers, args)
        
        results['mAP_qbe_%d' % (ot*100)] = mAP_qbe
        results['mR_qbe_%d' % (ot*100)] = mR_qbe
        results['mAP_qbs_%d' % (ot*100)] = mAP_qbs
        results['mR_qbs_%d' % (ot*100)] = mR_qbs
        
    return results

def average_precision_segfree(res, t, o, sinds, n_relevant, ot):
    """
    Computes the average precision
    
    res: sorted list of labels of the proposals, aka the results of a query.
    t: transcription of the query
    o: overlap matrix between the proposals and gt_boxes.
    sinds: The gt_box with which the proposals overlaps the most.
    n_relevant: The number of relevant retrievals in ground truth dataset
    ot: overlap_threshold
    """
    correct_label = res == t
    
    #The highest overlap between a proposal and a ground truth box
    tmp = []
    covered = []
    #TODO: this shouldn't really happen very often, check if possible at all, potential speed up
    for i in range(len(res)):
        if sinds[i] not in covered: #if a ground truth box has been covered, mark proposal as irrelevant
            tmp.append(o[i])
            if o[i] >= ot and correct_label[i]:
                covered.append(sinds[i])
        else:
            tmp.append(0.0)
            
#    for i in range(len(res)):
#        tmp.append(o[i, sinds[i]])
    
    tmp = np.array(tmp)
#    tmp = o
    relevance = correct_label * (tmp >= ot)
    covered = np.unique(sinds[relevance])
    rel_cumsum = np.cumsum(relevance, dtype=float)
    precision = rel_cumsum / np.arange(1, relevance.size + 1)
    if n_relevant > 0:
        ap = (precision * relevance).sum() / n_relevant
    else:
        ap = 0.0
        
    return ap, covered

def worker(arg):
    dists, t, db_targets, joint_boxes, nms_overlap, max_overlaps, amax_overlaps, gt_targets, ot, num_workers = arg
    count = np.sum(db_targets == t)
    if count == 0: #i.e., we have missed this word completely
        return 0.0, 0.0
    
    sim = 1 - dists
    dets = np.hstack((joint_boxes, sim[:, np.newaxis]))
    pick = box_utils.nms_np(dets, nms_overlap)
    dists = dists[pick]
    I = np.argsort(dists)
    res = db_targets[pick][I]   #Sort results after distance to query image
    o = max_overlaps[pick][I]
    sinds = amax_overlaps[pick][I]
    n_relevant = np.sum(gt_targets == t)
    ap, covered = average_precision_segfree(res, t, o, sinds, n_relevant, ot)
    r = float(np.unique(covered).shape[0]) / n_relevant
    return ap, r
    

def mAP_parallel(dists, qtargets, db_targets, gt_targets, joint_boxes, 
                 max_overlaps, amax_overlaps, nms_overlap, ot, num_workers, opt):
    args = [(d, t, db_targets, joint_boxes, nms_overlap, max_overlaps, 
             amax_overlaps, gt_targets, ot, num_workers) for d, t in zip(dists, qtargets)]

    if num_workers > 1:
        with closing(PyPool(num_workers)) as p:
            res = p.map(worker, args)
             
    else:
        res = [worker(arg) for arg in args]
               
    res = np.array(res)
    return np.mean(res, axis=0)
