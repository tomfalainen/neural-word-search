#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:25:01 2017

@author: tomas
"""
import torch

def xcycwh_to_x1y1x2y2_batch(boxes):
    """
    Boxes are N x B x 4
    """
    
    minibatch = True
    if boxes.ndimension() == 2:
        minibatch = False
        boxes = boxes.view(1, boxes.size(1), 4)
  
    xc = boxes[:, :, 0]
    yc = boxes[:, :, 1]
    w = boxes[:, :, 2]
    h = boxes[:, :, 3]
    
    wh = w / 2.0
    hh = h / 2.0
    x1 = xc - wh
    x2 = xc + wh
    y1 = yc - hh
    y2 = yc + hh
    ret = torch.stack((x1, y1, x2, y2), dim=2)
    
    if not minibatch:
        ret = ret.view(boxes.size(1), 4)
  
    return ret    

def x1y1x2y2_to_xcycwh_batch(boxes):
    """
    Boxes are N x 4
    """
    minibatch = True
    if boxes.ndimension() == 2:
        minibatch = False
        boxes = boxes.view(1, boxes.size(1), 4)
  
    x1 = boxes[:, :, 0]
    y1 = boxes[:, :, 1]
    x2 = boxes[:, :, 2]
    y2 = boxes[:, :, 3]
    
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    ret = torch.stack((xc, yc, w, h), dim=2)
    
    if not minibatch:
        ret = ret.view(boxes.size(1), 4)
    return ret

def xcycwh_to_x1y1x2y2(boxes):
    """
    Boxes are N x 4
    """
    xc = boxes[:, 0]
    yc = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    #First version, probably correct    
    wh = w / 2.0
    hh = h / 2.0
    
    x1 = xc - wh
    x2 = xc + wh
    y1 = yc - hh
    y2 = yc + hh
    
    ret = torch.stack((x1, y1, x2, y2), dim=1)
    return ret    

def x1y1x2y2_to_xcycwh(boxes):
    """
    Boxes are N x 4
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    ret = torch.stack((xc, yc, w, h), dim=1)
    return ret

def clip_boxes(boxes_in, bounds, format):
    if boxes_in.dim() == 3:
        boxes = boxes_in.view(-1, 4)
    else:
        boxes = boxes_in
    
    if format == 'x1y1x2y2':
        boxes_clipped = boxes.clone()
    elif format == 'xcycwh':
        boxes_clipped = xcycwh_to_x1y1x2y2(boxes)
    # elif format == 'xywh':
        # boxes_clipped = xywh_to_x1y1x2y2(boxes)
    else:
        raise ValueError('Unrecognized box format %s' % format)

    # Now we can actually clip the boxes
    boxes_clipped[:, 0] = boxes_clipped[:, 0].clamp(bounds['x_min'], bounds['x_max'] - 1)
    boxes_clipped[:, 1] = boxes_clipped[:, 1].clamp(bounds['y_min'], bounds['y_max'] - 1)
    boxes_clipped[:, 2] = boxes_clipped[:, 2].clamp(bounds['x_min'] + 1, bounds['x_max'])
    boxes_clipped[:, 3] = boxes_clipped[:, 3].clamp(bounds['y_min'] + 1, bounds['y_max'])

    #For tests use this version
#    boxes_clipped[:, 0] = boxes_clipped[:, 0].clamp(bounds['x_min'], bounds['x_max'])
#    boxes_clipped[:, 1] = boxes_clipped[:, 1].clamp(bounds['y_min'], bounds['y_max'])
#    boxes_clipped[:, 2] = boxes_clipped[:, 2].clamp(bounds['x_min'], bounds['x_max'])
#    boxes_clipped[:, 3] = boxes_clipped[:, 3].clamp(bounds['y_min'], bounds['y_max'])

    validx = torch.gt(boxes_clipped[:,2], boxes_clipped[:,0]).byte()
    validy = torch.gt(boxes_clipped[:,3], boxes_clipped[:,1]).byte()
    valid = torch.gt(validx * validy, 0) # logical and operator

    # Convert to the same format as the input
    if format == 'xcycwh':
        boxes_clipped = x1y1x2y2_to_xcycwh(boxes_clipped)
  
    # Conver to the same shape as the input
    return boxes_clipped.view_as(boxes_in), valid

def nms_np(boxes, overlap):
    return nms(torch.from_numpy(boxes), overlap).cpu().numpy()

def nms(boxes, overlap, max_boxes=None):
    if isinstance(boxes, list) or isinstance(boxes, tuple):
        # -- unpack list into boxes array and scores array
        s = boxes[2]
        boxes = boxes[1]
    else:
        # -- boxes is a tensor and last column are scores
        s = boxes[:, -1]
  
    if boxes.numel() == 0:
        return torch.zeros(0)
  
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0 )

    vals, I = s.sort(0)
    pick = torch.zeros(s.size()).long()
    if boxes.is_cuda:
        pick = pick.cuda()
        
    counter = 0
    while (max_boxes == None or counter < max_boxes) and I.numel() > 0:
        last = I.size(0) - 1
        i = I[last]
        pick[counter] = i
        if last == 0:
            break
        
        counter += 1
    
        I = I[:last]

        # Compute IoU between current box and all boxes
        xx1 = torch.clamp(x1, min=x1[i])
        xx2 = torch.clamp(x2, max=x2[i])
        yy1 = torch.clamp(y1, min=y1[i])
        yy2 = torch.clamp(y2, max=y2[i])
        
        w = torch.clamp((xx2 - xx1) + 1.0, min=0)
        h = torch.clamp((yy2 - yy1) + 1.0, min=0)
        inter = w * h
        union = (area + area[i]) - inter
        iou = inter / union
    
        # Figure out which boxes have IoU below the threshold with the current box;
        # since we only really need to know IoU between the current box and the
        # boxes specified by I, pick those elements out.
        mask = torch.gather(iou.le(overlap).byte(), dim=0, index=I)

        I = I[mask]
  
    pick = pick[:counter]
    return pick