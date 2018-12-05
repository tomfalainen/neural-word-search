#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:01:57 2017

@author: tomas
"""

import box_sampler
import torch
import utils

class BoxSamplerHelper(torch.nn.Module):
    def __init__(self, opt):
        super(BoxSamplerHelper, self).__init__()
        if opt.has_key('box_sampler'):
            self.box_sampler = opt['box_sampler'] #For testing
        else:
            self.box_sampler = box_sampler.BoxSampler(opt)
            
        self.contrastive_loss = opt['contrastive_loss']
        self.return_index = utils.getopt(opt, 'return_index', False)
            
    def setBounds(self, bounds):
        self.box_sampler.setBounds(bounds)
        
    """
      Input:
      
      List of two lists. The first list contains data about the input boxes,
      and the second list contains data about the target boxes.

      The first element of the first list is input_boxes, a Tensor of shape (N, B1, 4)
      giving coordinates of the input boxes in (xc, yc, w, h) format.

      All other elements of the first list are tensors of shape (N, B1, Di) parallel to
      input_boxes; Di can be different for each element.

      The first element of the second list is target_boxes, a Tensor of shape (N, B2, 4)
      giving coordinates of the target boxes in (xc, yc, w, h) format.

      All other elements of the second list are tensors of shape (N, B2, Dj) parallel
      to target_boxes; Dj can be different for each Tensor.

      
      Returns a list of three lists:

      The first list contains data about positive input boxes. The first element is of
      shape (P, 4) and contains coordinates of positive boxes; the other elements
      correspond to the additional input data about the input boxes; in particular the
      ith element has shape (P, Di).

      The second list contains data about target boxes corresponding to positive
      input boxes. The first element is of shape (P, 4) and contains coordinates of
      target boxes corresponding to sampled positive input boxes; the other elements
      correspond to the additional input data about the target boxes; in particular the
      jth element has shape (P, Dj).

      The third list contains data about negative input boxes. The first element is of
      shape (M, 4) and contains coordinates of negative input boxes; the other elements
      correspond to the additional input data about the input boxes; in particular the
      ith element has shape (M, Di).
    """        
    
    def forward(self, input):
        input_data = input[0]
        target_data = input[1]
        input_boxes = input_data[0]
        target_boxes = target_data[0]
        N = input_boxes.size(0)
        assert N == 1, 'Only minibatches of 1 are supported'
        
         # Run the sampler to get the indices of the positive and negative boxes
        idxs = self.box_sampler.forward((input_boxes.data, target_boxes.data))
        pos_input_idx = idxs[0]
        pos_target_idx = idxs[1]
        neg_input_idx = idxs[2]
        
        n = pos_target_idx.size(0)
        y = torch.autograd.Variable(torch.ones(n), requires_grad=False)
        y = y.float()
        if input_boxes.data.is_cuda:
            y = y.cuda()
        
        # Inject mismatching pairs for the cosine embedding loss here, and save which pairs are mismatched
        if self.contrastive_loss:
            frac = 0.2 #The fraction of how many negative pairs are injected
            z = torch.rand(n).lt(frac)
            if input_boxes.data.is_cuda:
                z = z.cuda()
                
            y[z] = -1
            p = torch.ones(target_boxes.size(1))
    
            # Randomly select other word embeddings from the same page.
            modified_pos_target_idx = pos_target_idx.clone()
            if z.sum() > 0:
                modified_pos_target_idx[z] = torch.multinomial(p, z.sum(), replacement=True).type(pos_target_idx.type())

        # Resize the output. We need to allocate additional tensors for the
        # input data and target data, then resize them to the right size.
        num_pos = pos_input_idx.size(0)
        num_neg = neg_input_idx.size(0)
            
        # -- Now use the indeces to actually copy data from inputs to outputs
        pos, target, neg = [], [], []
        for i in range(len(input_data)):
            d = input_data[i]
            D = d.size(2)
            pos.append(d[:, pos_input_idx].view(num_pos, D))
            neg.append(d[:, neg_input_idx].view(num_neg, D))
            
        for i in range(len(target_data)):
            d = target_data[i]
            D = d.size(2) 
            #For the ground truth embeddings, inject the mismatching pairs
            if self.contrastive_loss:
                if i == 1:
                    target.append(d[:, modified_pos_target_idx].view(num_pos, D))
                else:#For the ground truth boxes, use the correct pos_target_idx
                    target.append(d[:, pos_target_idx].view(num_pos, D))
            else:
                target.append(d[:, pos_target_idx].view(num_pos, D))
                
        output = (pos, target, neg, y)
        if self.return_index:
            output += (idxs,)
        return output