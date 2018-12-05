import torch
import batch_bilinear_sampler_bhwd
import box_to_affine

"""
BilinearRoiPooling is a layer that uses bilinear sampling to pool featurs for a
region of interest (RoI) into a fixed size.

The constructor takes inputs height and width, both integers giving the size to
which RoI features should be pooled. For example if RoI feature maps are being
fed to VGG-16 fully connected layers, then we should have height = width = 7.

WARNING: The bounding box coordinates given in the forward pass should be in
the coordinate system of the input image used to compute the feature map, NOT
in the coordinate system of the feature map. To properly compute the forward
pass, the module needs to know the size of the input image; therefore the method
setImageSize(image_height, image_width) must be called before each forward pass.

Inputs:
- feats: Tensor of shape (C, H, W) giving a convolutional feature map.
- boxes: Tensor of shape (B, 4) giving bounding box coordinates in
         (xc, yc, w, h) format; the bounding box coordinates are in
         coordinate system of the original image, NOT the convolutional
         feature map.

Return:
- roi_features: Tensor of shape (B, C, output_height, output_width)
"""

class BilinearRoiPooling(torch.nn.Module):
    def __init__(self, output_height, output_width):
        super(BilinearRoiPooling, self).__init__()
        self.output_height = output_height
        self.output_width = output_width
  
        #  -- box_to_affine converts boxes of shape (B, 4) to affine parameter
        #  -- matrices of shape (B, 2, 3); on each forward pass we need to call
        #  -- box_to_affine.setSize() to set the size of the input image.
        self.box_to_affine = box_to_affine.BoxToAffine()
        self.sampler = batch_bilinear_sampler_bhwd.BatchBilinearSamplerBHWD()
        self.called_forward = False
        self.called_backward = False

    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self.called_forward = False
        self.called_backward = False

    def forward(self, input):
        assert self.image_height and self.image_width and not self.called_forward, \
             'Must call setImageSize before each forward pass'
        
        feats, boxes = input
        B = boxes.size(0)
        C, H, W = feats.size()
        self.box_to_affine.setSize(self.image_height, self.image_width)
        aff = self.box_to_affine(boxes)
        s = torch.Size([B, C, self.output_height, self.output_width])
        grids = torch.nn.functional.affine_grid(aff, s)
        output = self.sampler.forward((feats, grids))
        self.called_forward = True
        return output