import torch
import torch.nn as nn
import easydict

#Load submodules
from box_sampler_helper import BoxSamplerHelper
from bilinear_roi_pooling import BilinearRoiPooling

import utils

class LocalizationLayer(nn.Module):
    def __init__(self, opt):
        super(LocalizationLayer, self).__init__()
        self.opt = easydict.EasyDict()
        self.opt.input_dim = utils.getopt(opt, 'input_dim')
        self.opt.output_size = utils.getopt(opt, 'output_size')
        self.opt.sampler_batch_size = utils.getopt(opt, 'sampler_batch_size')
        self.opt.sampler_high_thresh = utils.getopt(opt, 'sampler_high_thresh')
        self.opt.sampler_low_thresh = utils.getopt(opt, 'sampler_low_thresh')
        self.opt.train_remove_outbounds_boxes = utils.getopt(opt, 'train_remove_outbounds_boxes', 1)
        self.opt.contrastive_loss = utils.getopt(opt, 'contrastive_loss')

        sampler_opt = {'batch_size':self.opt.sampler_batch_size,
               'low_thresh':self.opt.sampler_low_thresh,
               'high_thresh':self.opt.sampler_high_thresh,
               'contrastive_loss':self.opt.contrastive_loss}
        
        self.box_sampler_helper = BoxSamplerHelper(sampler_opt)
        self.roi_pooling = BilinearRoiPooling(self.opt.output_size[0], self.opt.output_size[1])

        self.image_height = None
        self.image_width = None
        self._called_forward_size = False
        self._called_backward_size = False
        
    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self._called_forward_size = False
        self._called_backward_size = False

    def setTestArgs(self, args={}):
        self.test_clip_boxes = utils.getopt(args, 'clip_boxes', True)
        self.test_nms_thresh = utils.getopt(args, 'nms_thresh', 0.7)
        self.test_max_proposals = utils.getopt(args, 'max_proposals', 300)
        
    def gt_sample(self, gt_boxes, gt_embeddings, num_pos=None):
        total_pos = gt_boxes.size(1)
        pos_mask = torch.ones(total_pos)
        pos_mask_nonzero = pos_mask.nonzero().view(-1)
        if not num_pos:
            num_pos = self.opt.sampler_batch_size
        pos_p = torch.ones(total_pos)
        pos_sample_idx = torch.multinomial(pos_p, num_pos, replacement=False)
        pos_input_idx = pos_mask_nonzero.index(pos_sample_idx)
        y = torch.autograd.Variable(torch.ones(num_pos).long())
        
        if gt_boxes.is_cuda:
            pos_input_idx = pos_input_idx.cuda()    
            y = y.cuda()
        
        boxes_out = gt_boxes[:, pos_input_idx].view(num_pos, 4)
        D = gt_embeddings.size(2)
        embeddings_out = gt_embeddings[:, pos_input_idx].view(num_pos, D)
        out = (boxes_out, embeddings_out, y)
        return out
        
    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            return self._forward_test(input)
        
    def _forward_train(self, input):
        cnn_features, gt_boxes, gt_embeddings, proposals = input
        
#        -- Make sure that setImageSize has been called
        assert self.image_height and self.image_width and not self._called_forward_size, \
         'Must call setImageSize before each forward pass'
        self._called_forward_size = True

        N = cnn_features.size(0)
        assert N == 1, 'Only minibatches with N = 1 are supported'
        B1 = gt_boxes.size(1)
        assert gt_boxes.dim() == 3 and gt_boxes.size(0) == N and gt_boxes.size(2) == 4, \
         'gt_boxes must have shape (N, B1, 4)'
        assert gt_embeddings.dim() == 3 and gt_embeddings.size(0) == N and gt_embeddings.size(1) == B1, \
         'gt_embeddings must have shape (N, B1, L)'

        # Run the sampler forward
        sampler_in = (proposals,)
        sampler_out = self.box_sampler_helper.forward((sampler_in, (gt_boxes, gt_embeddings)))

        # Unpack pos data
        pos_data, pos_target_data, neg_data, y = sampler_out
        pos_boxes = pos_data[0]
        num_pos = pos_boxes.size(0)

        # Unpack target data
        pos_target_boxes, pos_target_labels = pos_target_data
 
        # Unpack neg data (only scores matter)
        neg_boxes = neg_data[0]

        #This happens sometimes, especially before model has converged somewhat
        #We fix it by adding some boxes we know to be positive from the ground truth
        # and remove the corresponding amount of negative boxes.
        if num_pos <= 1:
            N = 64
            gt_train_boxes, gt_train_embeddings, gt_y = self.gt_sample(gt_boxes, gt_embeddings, num_pos=N)
            pos_boxes = torch.cat((pos_boxes, gt_train_boxes), dim=0)
            y = torch.cat((y, gt_y), dim=0)
            pos_target_boxes = torch.cat((pos_target_boxes, gt_train_boxes), dim=0)
            pos_target_labels = torch.cat((pos_target_labels, gt_train_embeddings), dim=0)
            neg_boxes = neg_boxes[:-N]

        # Concatentate pos_boxes and neg_boxes into roi_boxes
        roi_boxes = torch.cat((pos_boxes, neg_boxes), dim=0)

        # Run the RoI pooling forward for roi_boxes
        self.roi_pooling.setImageSize(self.image_height, self.image_width)
        roi_features = self.roi_pooling.forward((cnn_features[0], roi_boxes))
        output = (roi_features, roi_boxes, pos_target_boxes, pos_target_labels, y)
        return output
    
    #Clamp parallel arrays only to valid boxes (not oob of the image)
    def clamp_data(self, data, valid):
        #data should be 1 x kHW x D
        #valid is byte of shape kHW
        assert data.size(0) == 1, 'must have 1 image per batch'
        assert data.dim() == 3
        mask = valid.view(1, -1, 1).expand_as(data)
        return data[mask].view(1, -1, data.size(2))
        
    def eval_boxes(self, input):
        """
        performs bilinear interpolation on the given boxes on the input features.
        Useful for when using external proposals or ground truth boxes
        
        Boxes should be in xc, yc, w, h format
        """
        cnn_features, boxes = input

        # Use roi pooling to get features for boxes
        self.roi_pooling.setImageSize(self.image_height, self.image_width)
        features = self.roi_pooling.forward((cnn_features[0], boxes))
        return features