import torch
import torch.nn.functional as F

"""
  BatchBilinearSamplerBHWD efficiently performs bilinear sampling to pull out
  multiple patches from a single input image.

  Inputs:
  - inputImages: Tensor of shape (H, W, C)
  - grids: Tensor of shape (N, HH, WW, 2)

  Output:
  - Tensor of shape (N, HH, WW, C) which is the result of applying each
    sampling grid to the input image.

"""

class BatchBilinearSamplerBHWD(torch.nn.Module):
    def check(self, input):
        inputImages = input[0]
        grids = input[1]
    
        assert inputImages.ndimension()==4
        assert grids.ndimension()==4
        assert inputImages.size(0)==grids.size(0) # batch
        assert grids.size(3)==2 # coordinates


    def forward(self, input):
        #  -- inputImages should be C x H x W
        #  -- grids should be B x HH x WW x 2
        inputImages, grids = input[0], input[1]
  
        assert grids.dim() == 4
        B = grids.size(0)
  
        assert inputImages.dim() == 3
        C, H, W = inputImages.size(0), inputImages.size(1), inputImages.size(2)
        inputImageView = torch.unsqueeze(inputImages, dim=0).expand(B, C, H, W)
        self.check((inputImageView, grids))
        output = F.grid_sample(inputImageView, grids)
        return output
