#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:59:42 2017

@author: tomas
"""

import torch
from torch.autograd import Variable

"""
One-vs-all logistic loss; each example has a single positive class.

On the forward pass we take:
- input: Tensor of shape (N, C) giving scores for C classes for each
  of N examples.
- target: LongTensor of shape (N) giving labels for each of the N
  examples; each element is an integer in the range [0, C] with the
  interpretation that target[i] = 0 means that input[i] is a negative
  example for all classes; if target[i] = c > 0 then input[i] is a positive
  example for class c and a negative example for all other classes.

The amounts to evaluating the binary logistic loss for each element of the
(N, C) array of scores. For an element x = scores[{i, j}], its binary label
is y = 1 if target[i] = j and y = 0 otherwise. The binary logistic loss is
given by:

loss(x, y) = log(1 + exp(-x))        if y == 1
             log(1 + exp(-x)) + x    if y == 0

You can derive this as KL(target, predicted) where target and predicted are
distributions over two classes (positive and negative), the target
distribution is

P(pos) = y
P(neg) = 1 - y

and the predicted distribution is

P(pos) = 1 / (1 + exp(-x))
P(neg) = exp(-x) / (1 + exp(-x)))

To improve numeric stability, we make use of the fact that for all a,

log(1 + exp(-x)) = log(exp(a) + exp(a - x)) - a

In practice we choose a = min(0, x) to make sure that all exponents
are negative; this way we won't have overflow resulting in inf, but
we may have underflow resulting in 0 which is preferable.
"""

class LogisticLoss(torch.nn.Module):
   def forward(self, input, target):
        """
          Inputs:
          - input: N tensor of class scores
          - target: N LongTensor giving ground-truth for elements of inputs;
                    each element should be an integer in the range [0, C];
                    if target[i] == 0 then input[i] should be negative for all classes.
        """
        gpu = input.is_cuda
        ones = Variable(torch.zeros(input.size()))
        if gpu:
            ones = ones.cuda()
        a = torch.min(input, ones)

        log_den = torch.log(torch.exp(a) + torch.exp(a - input)) - a
        mask = torch.eq(target, 0)
        
        losses = log_den + input * mask.float()
        output = torch.mean(losses)
        return output
