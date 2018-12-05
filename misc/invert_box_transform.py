#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:44:30 2017

@author: tomas
"""
import torch

class InvertBoxTransform(torch.nn.Module):
    """
    Given anchor boxes and target boxes, compute the box transform parameters that
    would be needed to transform the anchors into the targets. This is an inverse
    for ApplyBoxTransform.

    Inputs:
    - anchor_boxes: Tensor of shape (B, 4) giving coordinates for B anchor boxes in
      (xc, yc, w, h) format.
    - target_boxes: Tensor of shape (B, 4) giving coordinates for B target boxes in
      (xc, yc, w, h) format.

    Outputs:
    - trans: Tensor of shape (B, 4) giving box transforms in the format
      (tx, ty, tw, th) such that applying trans[i] to anchor_boxes[i] gives
      target_boxes[i].
    """

    def forward(self, input):
        anchor_boxes, target_boxes = input[0], input[1]

        xa = anchor_boxes[:, 0]
        ya = anchor_boxes[:, 1]
        wa = anchor_boxes[:, 2]
        ha = anchor_boxes[:, 3]

        xt = target_boxes[:, 0]
        yt = target_boxes[:, 1]
        wt = target_boxes[:, 2]
        ht = target_boxes[:, 3]

        px = (xt - xa) / wa
        py = (yt - ya) / ha
        pw = (wt / wa).log()
        ph = (ht / ha).log()
        output = torch.stack((px, py, pw, ph), dim=1)
        return output