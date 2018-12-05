#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:00:04 2017

@author: tomas
"""
import torch
class ApplyBoxTransform(torch.nn.Module):
    """
    Apply adjustments to bounding boxes for bounding box regression, with
    backpropagation both into box offsets and box positions.

    We use the same parameterization for box regression as R-CNN:

    Given a bounding box with center (xa, ya), width wa, and height ha,
    and given offsets (tx, ty, tw, th), we compute the new bounding box
    (x, y, w, h) as:

    x = tx * wa + xa
    y = ty * ha + ya
    w = wa * exp(tw)
    h = ha * exp(th)

    This parameterization is nice because the identity transform is (0, 0, 0, 0).

    Given gradients (dx, dy, dw, dh) on the output the gradients on the inputs are

    dtx = wa * dx
    dty = ha * dy
    dtw = dw * wa * exp(tw) = dw * w
    dth = dh * ha * exp(th) = dh * h

    dxa = dx
    dya = dy
    dwa = dx * tx + dw * exp(tw)
    dha = dy * ty + dh * exp(th)


    Module input: A list of
    - boxes: Tensor of shape (D1, D2, ..., 4) giving coordinates of boxes in
             (xc, yc, w, h) format.
    - trans: Tensor of shape (D1, D2, ..., 4) giving box transformations in the form
             (tx, ty, tw, th)

    Module output:
    - Tensor of shape (D1, D2, ..., 4) giving coordinates of transformed boxes in
      (xc, yc, w, h) format. Output has same shape as input.

    """

    def forward(self, input):
        boxes, trans = input[0], input[1]

        assert boxes.size(-1) == 4, 'Last dim of boxes must be 4'
        assert trans.size(-1) == 4, 'Last dim of trans must be 4'
        boxes_view = boxes.contiguous().view(-1, 4)
        trans_view = trans.contiguous().view(-1, 4)

        xa = boxes_view[:, 0]
        ya = boxes_view[:, 1]
        wa = boxes_view[:, 2]
        ha = boxes_view[:, 3]

        tx = trans_view[:, 0]
        ty = trans_view[:, 1]
        tw = trans_view[:, 2]
        th = trans_view[:, 3]

        x = tx * wa + xa
        y = ty * ha + ya
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        output = torch.stack((x, y, w, h), dim=1)
        output = output.view(boxes.size())
        return output