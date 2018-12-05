import torch

"""
Convert bounding box coordinates to affine parameter matrices that can be used
to generate sampling grids for bilinear interpolation.

Input: Tensor of shape (B, 4) giving bounding box coordinates in (xc, yc, w, h)
       format.

Output: Tensor of shape (B, 2, 3) giving affine parameter matrices for boxes.

If the input image has height H and width W, then for each box (xc, yc, w, h)
we want to generate the following 2 x 3 affine transform matrix:

 [   h             2 * yc - H - 1 ]
 [  ---      0     -------------- ]
 [   H                  H - 1     ]
 [                                ]
 [           w     2 * xc - W - 1 ]
 [   0      ---    -------------- ]
 [           W          W - 1     ]

This looks funny because the affine transform matrices are expected to work on
normalized coordinates in the range [-1, 1] x [1, 1] rather than image space
coordinates in the range [1, W] x [1, H]. The (1, 3) and (2, 3) elements of the
matrix give the center of the bounding box in the normalized coordinate system,
and the (1, 1) and (2, 2) elements of the matrix give the size of the box in
the normalized coordinate system.

The matrix defines a mapping from the (normalized) output coordinate system to
the (normalized) input coordinate system so (0, 0) maps to the box center and
(+/- 1, +/- 1) map to the four corners of the box. This transform is achieved
by multiplying the parameter matrix on the right by the column vector
(y, x, 1).

NOTE: In the Spatial Transformer Networks paper, the parameter matrix expects
to multiply the vector (x, y, 1) but in qassemoquab/stnbhwd the
AffineGridGenerator expects (y, x, 1). This inconsistency is tough to catch
from numeric unit tests alone, so there is an iTorch notebook with visual
sanity check tests to make sure that bounding boxes in image coordinates are
selecting the correct portions of the image.

Thanks to normalized coordinates, H and W can be the size of the input image
but the affine parameter matrix can be used to sample from convolutional layers.
This works because the coordinate system of the conv feature map is shifted and
scaled relative to the coordinate system of the input image, which implies that
the normalized coordinate systems for the image and feature map are the same.

NOTE: This module will frequently be used with different underlying image sizes
at each iteration; for this reason the setSize method should be called before
each call to forward.
"""

class BoxToAffine(torch.nn.Module):
    def __init__(self):
        super(BoxToAffine, self).__init__()
        self.H = None
        self.W = None

    def setSize(self, H, W):
        self.H = H
        self.W = W

    def forward(self, input):
        assert input.dim() == 2, 'Expected 2D input'
        B = input.size(0)
        assert input.size(1) == 4, 'Expected input of shape B x 4'
        assert self.H and self.W, 'Need to call setSize before calling forward'

        xc = input[:, 0]
        yc = input[:, 1]
        w = input[:, 2]
        h = input[:, 3]
        
        th13 = torch.div((yc * 2) - 1 - self.H, self.H - 1)
        th23 = torch.div((xc * 2) - 1 - self.W, self.W - 1)
        th22 = torch.div(w, self.W)
        th11 = torch.div(h, self.H)
        th12 = torch.autograd.Variable(torch.zeros_like(xc.data))
        th21 = torch.autograd.Variable(torch.zeros_like(xc.data))
#        output = torch.autograd.Variable(torch.zeros(B, 2, 3).type_as(input.data), requires_grad=True)
#        output[:, 0, 0] = th11
#        output[:, 0, 2] = th13
#        output[:, 1, 1] = th22
#        output[:, 1, 2] = th23
#        r1 = torch.stack((th11, th12, th13), dim=1).view(B, 1, 3)
#        r2 = torch.stack((th21, th22, th23), dim=1).view(B, 1, 3)
        
        #W, H are flipped compared to torch version
        r1 = torch.stack((th22, th12, th23), dim=1).view(B, 1, 3)
        r2 = torch.stack((th21, th11, th13), dim=1).view(B, 1, 3)
        
        output = torch.cat((r1, r2), dim=1)
        return output
