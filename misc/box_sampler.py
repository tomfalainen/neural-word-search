
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:57:28 2017

@author: tomas
"""
import torch
import torch.nn as nn
import utils
import box_utils
import boxIoU

class BoxSampler(nn.Module):
    def __init__(self, opt):
        super(BoxSampler, self).__init__()
        self.opt = {}
        self.low_thresh = utils.getopt(opt, 'low_thresh', 0.4)
        self.high_thresh = utils.getopt(opt, 'high_thresh', 0.75)
        self.batch_size = utils.getopt(opt, 'batch_size', 256)
        self.debug = utils.getopt(opt, 'debug', False)
        
        self.iou = boxIoU.BoxIoU()
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
      
    def unpack_dims(self, input_boxes, target_boxes):
        N, B1 = input_boxes.size(0), input_boxes.size(1)
        B2 = target_boxes.size(1)
        assert N == 1, "Only 1-element minibatches are supported"
        assert input_boxes.size(2) == 4 and target_boxes.size(2) == 4
        assert target_boxes.size(0) == N
        return N, B1, B2
        
    def setBounds(self, bounds):
        self.x_min = utils.getopt(bounds, 'x_min', None)
        self.x_max = utils.getopt(bounds, 'x_max', None)
        self.y_min = utils.getopt(bounds, 'y_min', None)
        self.y_max = utils.getopt(bounds, 'y_max', None)
        
    def forward(self, input):
        input_boxes, target_boxes = input
        N, B1, B2 = self.unpack_dims(input_boxes, target_boxes)
        #For now, only support batch size of 1
        input_boxes = input_boxes.view(B1, 4)
        target_boxes = target_boxes.view(B2, 4)
        ious = boxIoU.boxIoU(input_boxes, target_boxes).view(N, B1, B2) # N x B1 x B2
        input_max_iou, input_idx = torch.max(ious, dim=2) 
        input_max_iou = input_max_iou.view(N, B1) #N x B1
        target_max_iou, target_idx = torch.max(ious, dim=1) 
        target_max_iou = target_max_iou.view(N, B2) #N x B2
        pos_mask = torch.gt(input_max_iou, self.high_thresh) # N x B1
        neg_mask = torch.lt(input_max_iou, self.low_thresh) # N x B1
        
        if self.x_min and self.y_min and self.x_max and self.y_max:
            boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)
            x_min_mask = torch.lt(boxes_x1y1x2y2[:, 0], self.x_min).byte()
            y_min_mask = torch.lt(boxes_x1y1x2y2[:, 1], self.y_min).byte()
            x_max_mask = torch.gt(boxes_x1y1x2y2[:, 2], self.x_max).byte()
            y_max_mask = torch.gt(boxes_x1y1x2y2[:, 3], self.y_max).byte()
            pos_mask[x_min_mask] = 0
            pos_mask[y_min_mask] = 0
            pos_mask[x_max_mask] = 0
            pos_mask[y_max_mask] = 0
            neg_mask[x_min_mask] = 0
            neg_mask[y_min_mask] = 0
            neg_mask[x_max_mask] = 0
            neg_mask[y_max_mask] = 0
            
        #    -- Count as positive each input box that has maximal IoU with each target box,
        #  -- even if it is outside the bounds or does not meet the thresholds.
        #  -- This is important since things will crash if we don't have at least one
        #  -- positive box.
        pos_mask = pos_mask.scatter_(1, target_idx, 1)
        neg_mask = neg_mask.scatter_(1, target_idx, 0)
        
        pos_mask = pos_mask.view(B1).byte()
        neg_mask = neg_mask.view(B1).byte()
        if neg_mask.sum() == 0:
            #    -- There were no negatives; this can happen if all input boxes are either:
            #    -- (1) An input box with maximal IoU with a target box
            #    -- (2) Out of bounds, therefore clipped
            #    -- (3) max IoU to all target boxes is in the range [low_thresh, high_thresh]
            #    -- This should be a pretty rare case, but we still need to handle it.
            #    -- Ideally this should do something like sort the non-positive in-bounds boxes
            #    -- by their max IoU to target boxes and set the negative set to be those with
            #    -- minimal IoU to target boxes; however this is complicated so instead we'll
            #    -- just sample from non-positive boxes to get negatives.
            #    -- We'll also log this event in the __GLOBAL_STATS__ table; if this happens
            #    -- regularly then we should handle it more cleverly.

            neg_mask = 1 - pos_mask # set neg_mask to inverse of pos_mask
            print "In box_sampler.py, no negatives."
        
        pos_mask_nonzero = pos_mask.nonzero().view(-1)
        neg_mask_nonzero = neg_mask.nonzero().view(-1)
        total_pos = pos_mask_nonzero.size(0)
        total_neg = neg_mask_nonzero.size(0)
        num_pos = min(self.batch_size / 2, total_pos)
        num_neg = self.batch_size - num_pos
        pos_p = torch.ones(total_pos)
        neg_p = torch.ones(total_neg)
        pos_sample_idx = torch.multinomial(pos_p, num_pos, replacement=False)
        neg_replace = total_neg < num_neg
#        if neg_replace:
#            print "In box_sampler.py, negatives with replacement"
            
        neg_sample_idx = torch.multinomial(neg_p, num_neg, replacement=neg_replace)

        if input_boxes.is_cuda:
            pos_sample_idx = pos_sample_idx.cuda()    
            neg_sample_idx = neg_sample_idx.cuda()
        
        if hasattr(self, 'debug_pos_sample_idx'):
            pos_sample_idx = self.debug_pos_sample_idx
  
        if hasattr(self, 'debug_neg_sample_idx'):
            neg_sample_idx = self.debug_neg_sample_idx
            
        if self.debug:
            self.pos_mask = pos_mask
            self.neg_mask = neg_mask

        pos_input_idx = pos_mask_nonzero[pos_sample_idx]
        pos_target_idx = input_idx.view(-1)[(pos_input_idx)].view(num_pos)
        neg_input_idx = neg_mask_nonzero[neg_sample_idx]
        output = (pos_input_idx, pos_target_idx, neg_input_idx)
        return output

        