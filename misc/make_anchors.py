import torch
"""
A module that constructs anchor positions. Given k anchor boxes with different 
widths and heights, we want to slide those anchors across every position of the 
input feature map and output the coordinates of all these anchors.

Note that this module does not actually use the input (only its size) so its
backward pass always computes zero.

The constructor takes the following arguments:

- x0, y0: Numbers giving coordinates of receptive field centers for upper left
  corner of inputs.
- sx, sy: Numbers giving horizontal and vertical stride between receptive field
  centers.
- anchors: Tensor of shape (2, k) giving width and height for each of k anchor
  boxes.

Input:
N x C x H x W array of features

Output:
N x 4k x H x W array of anchor positions; if you reshape the output to
N x k x 4 x H x W then along the 3rd dim we have (xc, yc, w, h) giving the 
coordinates of the anchor box at that location.
"""

class MakeAnchors(torch.nn.Module):
    def __init__(self, x0, y0, sx, sy, anchors, tunable_anchors):
        super(MakeAnchors, self).__init__()
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy
        self.tunable_anchors = tunable_anchors
        if self.tunable_anchors:
            self.anchors = torch.nn.Parameter(anchors.clone())
        else:
            self.anchors = anchors.clone()

    def forward(self, input):
        N, H, W = input.size(0), input.size(2), input.size(3)
        k = self.anchors.size(1)

        if isinstance(input, torch.autograd.Variable):
            x_centers = torch.arange(0, W).type_as(input.data)
            y_centers = torch.arange(0, H).type_as(input.data)
            gpu = input.data.is_cuda
        else:
            x_centers = torch.arange(0, W).type_as(input)
            y_centers = torch.arange(0, H).type_as(input)
            gpu = input.is_cuda
            
        x_centers = torch.autograd.Variable(x_centers)
        y_centers = torch.autograd.Variable(y_centers)
            
        x_centers = x_centers * self.sx + self.x0
        y_centers = y_centers * self.sy + self.y0
  
        xc = x_centers.view(1, 1, 1, W).expand(N, k, H, W)
        yc = y_centers.view(1, 1, H, 1).expand(N, k, H, W)

        if self.tunable_anchors:    
            w = self.anchors[0].view(1, k, 1, 1).expand(N, k, H, W)
            h = self.anchors[1].view(1, k, 1, 1).expand(N, k, H, W)
        else:
            anchors = torch.autograd.Variable(self.anchors, requires_grad=True)
            if gpu:
                anchors = anchors.cuda()
    
            w = anchors[0].view(1, k, 1, 1).expand(N, k, H, W)
            h = anchors[1].view(1, k, 1, 1).expand(N, k, H, W)

        output = torch.stack([xc, yc, w, h], dim=2).view(N, 4*k, H, W)
        return output
