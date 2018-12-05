import torch

"""
Input a tensor of shape N x (D * k) x H x W
Reshape and permute to output a tensor of shape N x (k * H * W) x D 
"""

class ReshapeBoxFeatures(torch.nn.Module):
    def __init__(self, k):
        super(ReshapeBoxFeatures, self).__init__()
        self.k = k

    def forward(self, input):
      N, H, W = input.size(0), input.size(2), input.size(3)
      D = input.size(1) / self.k
      k = self.k
      a = input.view(N, k, D, H, W).permute(0, 1, 3, 4, 2).contiguous()
      output = a.view(N, k * H * W, D)
      return output
