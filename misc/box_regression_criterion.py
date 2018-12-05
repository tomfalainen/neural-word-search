import torch.nn as nn
import torch

"""
A criterion for bounding box regression losses.

For bounding box regression, we always predict transforms on top of anchor boxes.
Instead of directly penalizing the difference between the ground-truth box and
predicted boxes, penalize the difference between the transforms and the optimal
transforms that would have converted the anchor boxes into the ground-truth boxes.

This criterion accepts as input the anchor boxes, transforms, and target boxes;
on the forward pass it uses the anchors and target boxes to compute target tranforms,
and returns the loss between the input transforms and computed target transforms.

Since we need the gradients for the targets as well, we need to implement our own 
smooth l1 loss since the criterions and functions of pytorch don't calculate 
gradients w.r.t. target variables.

Inputs:
- input: A list of:
  - anchor_boxes: Tensor of shape (B, 4) giving anchor box coords as (xc, yc, w, h)
  - transforms: Tensor of shape (B, 4) giving transforms as (tx, ty, tw, th)
- target_boxes: Tensor of shape (B, 4) giving target boxes as (xc, yc, w, h)
"""

class BoxRegressionCriterion(nn.Module):
    def __init__(self, w = 5e-5):
        super(BoxRegressionCriterion, self).__init__()
        self.w = w
        
    def inv(self, input):
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

    def smooth_l1_loss(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        x = torch.abs(input - target)
        mse_mask = torch.squeeze(x.lt(1))
        l1_mask = torch.squeeze(x.ge(1))
        
        use_mse = mse_mask.float().sum().data[0] > 0
        use_l1 = l1_mask.float().sum().data[0] > 0
        
        assert use_mse or use_l1, 'this should never happen, in box_regression_criterion.py'
        
        if use_mse:
            l2 = x * mse_mask.float()
            mse = 0.5 * torch.sum(torch.pow(l2, 2))
        else:
            mse = 0
        if use_l1:
            l1 = (x  - 0.5) * l1_mask.float()
            l1 = torch.sum(l1)    
        else:
            l1 = 0
        
        out = (l1 + mse) / input.nelement()
        return out

    def forward(self, input, target_boxes):
        anchor_boxes, transforms = input
        target_transforms = self.inv((anchor_boxes, target_boxes))

        #  -- DIRTY DIRTY HACK: Ignore loss for boxes whose transforms are too big
        mask = torch.gt(torch.abs(target_transforms).max(dim=1, keepdim=True)[0], 10)
        mask = torch.squeeze(mask)
        mask_sum = mask.sum()# / 4
        if mask_sum.data[0] > 0:
            mask_nonzero = torch.squeeze(mask.nonzero())
            transforms = transforms.clone()
            transforms.index_fill_(0, mask_nonzero, 0)
            target_transforms = target_transforms.clone()
            target_transforms.index_fill_(0, mask_nonzero, 0)
  
        #since we need the gradient w.r.t.  target transforms
        output = self.w * self.smooth_l1_loss(transforms, target_transforms)
        return output

