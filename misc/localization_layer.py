from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import easydict

#Load submodules
from reshape_box_features import ReshapeBoxFeatures
from make_anchors import MakeAnchors
from apply_box_transform import ApplyBoxTransform
from box_sampler_helper import BoxSamplerHelper
from bilinear_roi_pooling import BilinearRoiPooling
from box_regression_criterion import BoxRegressionCriterion
from invert_box_transform import InvertBoxTransform

import box_utils
import utils

class RPN(nn.Module):
    def __init__(self, opt):
        super(RPN, self).__init__()
        if isinstance(opt.anchors, torch.Tensor): #debug mode
            self.anchors = torch.Tensor(opt.anchors).t().clone()
        elif opt.anchors == 'original':
            self.anchors = torch.Tensor([
            [30, 20], [90, 20], [150, 20], [210, 20], [300, 20],
            [30, 40], [90, 40], [150, 40], [210, 40], [300, 40],
            [30, 60], [90, 60], [150, 60], [210, 60], [300, 60],
            ]).t().clone()
                   
        self.anchors = self.anchors * opt.anchor_scale
        self.num_anchors = self.anchors.size(1)
        self.std = opt.std
        self.zero_box_conv = opt.zero_box_conv
        s = 1
        self.pad = int(math.floor(opt.rpn_filter_size / 2))

        self.conv_layer = nn.Conv2d(opt.input_dim, opt.rpn_num_filters, 
                                    kernel_size=opt.rpn_filter_size, padding=self.pad)
        
        self.head_filter_size = 1
        self.box_conv_layer = nn.Conv2d(opt.rpn_num_filters, 4 * self.num_anchors, self.head_filter_size, stride=s)
        self.rpn_conv_layer = nn.Conv2d(opt.rpn_num_filters, 2 * self.num_anchors, self.head_filter_size, stride=s)
        self.reshape_box_features = ReshapeBoxFeatures(self.num_anchors)
        
        x0, y0, sx, sy = opt.field_centers
        self.make_anchors = MakeAnchors(x0, y0, sx, sy, self.anchors, opt.tunable_anchors)
        self.apply_box_transform = ApplyBoxTransform()
        
        self.box_conv_layer.weight.data.zero_()
        self.box_conv_layer.bias.data.zero_()
        self.w = opt.box_reg_decay
        
    def init_weights(self):
        self.conv_layer.weight.data.normal_(0, self.std)
        self.conv_layer.bias.data.zero_()
        
        if self.zero_box_conv:
            self.box_conv_layer.weight.data.zero_()
        else:
            self.box_conv_layer.weight.data.normal_(0, self.std)
        
        self.box_conv_layer.bias.data.zero_()
        self.rpn_conv_layer.weight.data.normal_(0, self.std)
        self.rpn_conv_layer.bias.data.zero_()
        
        
    def forward(self, feats):
        feats = F.relu(self.conv_layer(feats))
        
        #Box branch
        #Compute boxes
        bfeats = self.box_conv_layer(feats)
        act_reg = 0.5 * self.w * torch.pow(bfeats.norm(2), 2)
        
        #create anchors
        anchors = self.make_anchors(bfeats)
        anchors = self.reshape_box_features(anchors)
        
        #reshape transforms and apply to anchors
        trans = self.reshape_box_features(bfeats)
        boxes = self.apply_box_transform((anchors, trans))
        
        #Scores branch
        scores = self.rpn_conv_layer(feats)
        scores = self.reshape_box_features(scores)
        return (boxes, anchors, trans, scores), act_reg
        
class LocalizationLayer(nn.Module):
    def __init__(self, opt):
        super(LocalizationLayer, self).__init__()
        self.opt = easydict.EasyDict()
        self.opt.input_dim = utils.getopt(opt, 'input_dim')
        self.opt.output_size = utils.getopt(opt, 'output_size')

        # list x0, y0, sx, sy
        self.opt.field_centers = utils.getopt(opt, 'field_centers')

        self.opt.mid_box_reg_weight = utils.getopt(opt, 'mid_box_reg_weight')
        self.opt.mid_objectness_weight = utils.getopt(opt, 'mid_objectness_weight')

        self.opt.rpn_filter_size = utils.getopt(opt, 'rpn_filter_size', 3)
        self.opt.rpn_num_filters = utils.getopt(opt, 'rpn_num_filters', 256)
        self.opt.zero_box_conv = utils.getopt(opt, 'zero_box_conv', True)
        self.opt.std = utils.getopt(opt, 'std', 0.01)
        self.opt.anchor_scale = utils.getopt(opt, 'anchor_scale', 1.0)
        self.opt.anchors = utils.getopt(opt, 'anchors', 'original')

        self.opt.sampler_batch_size = utils.getopt(opt, 'sampler_batch_size', 256)
        self.opt.sampler_high_thresh = utils.getopt(opt, 'sampler_high_thresh', 0.75)
        self.opt.sampler_low_thresh = utils.getopt(opt, 'sampler_low_thresh', 0.4)
        self.opt.train_remove_outbounds_boxes = utils.getopt(opt, 'train_remove_outbounds_boxes', 1)
        self.opt.box_reg_decay = utils.getopt(opt, 'box_reg_decay', 5e-5)
        self.opt.tunable_anchors = utils.getopt(opt, 'tunable_anchors', False)
        self.opt.backprop_rpn_anchors = utils.getopt(opt, 'backprop_rpn_anchors', False)
        self.box_loss = utils.getopt(opt, 'wordness_loss')
        self.opt.contrastive_loss = utils.getopt(opt, 'contrastive_loss')

        self.stats = easydict.EasyDict()
        self.stats.losses = easydict.EasyDict()
        self.stats.vars = easydict.EasyDict()
        self.dtp_train = utils.getopt(opt, 'dtp_train', False)
        
        if self.dtp_train:
            self.opt.sampler_batch_size /= 2
        
        sampler_opt = {'batch_size':self.opt.sampler_batch_size,
                       'low_thresh':self.opt.sampler_low_thresh,
                       'high_thresh':self.opt.sampler_high_thresh,
                       'contrastive_loss':self.opt.contrastive_loss}
        
        debug_sampler = utils.getopt(opt, 'box_sampler', False)
        if debug_sampler != False:
            sampler_opt['box_sampler'] = debug_sampler
        
        self.rpn = RPN(self.opt)
        self.box_sampler_helper = BoxSamplerHelper(sampler_opt)
        self.roi_pooling = BilinearRoiPooling(self.opt.output_size[0], self.opt.output_size[1])
        self.invert_box_transform = InvertBoxTransform()
        
        # Construct criterions
        if self.opt.backprop_rpn_anchors:
            self.box_reg_loss = BoxRegressionCriterion(self.opt.mid_box_reg_weight)
        else:
            self.box_reg_loss = nn.SmoothL1Loss() # for RPN box regression

        self.box_scoring_loss = nn.CrossEntropyLoss()
            
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
        
    def init_weights(self):
        self.rpn.init_weights()
        
    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            return self._forward_test(input)
        
    def _forward_train(self, input):
        cnn_features, gt_boxes, gt_embeddings = input[0], input[1], input[2]
        if isinstance(cnn_features, Variable):
            gpu = cnn_features.data.is_cuda
        else:
            gpu = cnn_features.is_cuda
        
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

        #Run the RPN forward
        #(boxes, anchors, trans, scores)
        rpn_out, act_reg = self.rpn.forward(cnn_features)
        
        if self.opt.train_remove_outbounds_boxes == 1:
            bounds = {'x_min':0, 'y_min':0, 'x_max':self.image_width, 'y_max':self.image_height}
            self.box_sampler_helper.setBounds(bounds)

        sampler_out = self.box_sampler_helper.forward((rpn_out, (gt_boxes, gt_embeddings)))
        
        # Unpack pos data
        pos_data, pos_target_data = sampler_out[0], sampler_out[1]
        neg_data, y = sampler_out[2], sampler_out[3]
        pos_boxes, pos_anchors, pos_trans, pos_scores = pos_data

        # Unpack target data
        pos_target_boxes, pos_target_labels = pos_target_data
 
        # Unpack neg data (only scores matter)
        neg_boxes = neg_data[0]
        neg_scores = neg_data[3]
        
        num_pos, num_neg = pos_boxes.size(0), neg_scores.size(0)    
        
        # Compute objectness loss
        pos_labels = Variable(torch.zeros(num_pos).long())
        neg_labels = Variable(torch.ones(num_neg).long())
        if gpu:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
        
        obj_weight = self.opt.mid_objectness_weight
        obj_loss_pos = obj_weight * self.box_scoring_loss(pos_scores, pos_labels)
        obj_loss_neg = obj_weight * self.box_scoring_loss(neg_scores, neg_labels)
        self.stats.losses.obj_loss_pos = obj_loss_pos.data
        self.stats.losses.obj_loss_neg = obj_loss_neg.data

        if self.opt.backprop_rpn_anchors:
            reg_loss = self.box_reg_loss.forward((pos_anchors, pos_trans), pos_target_boxes)
        else:
            # Compute targets for RPN bounding box regression
            #detach since no gradients needed for targets
            pos_trans_targets = self.invert_box_transform.forward((pos_anchors, pos_target_boxes)).detach()
    
            # DIRTY DIRTY HACK: To prevent the loss from blowing up, replace boxes
            # with huge pos_trans_targets with ground-truth
            max_trans = torch.abs(pos_trans_targets).max(1)[0]
            max_trans_mask = torch.gt(max_trans, 10).view(-1, 1).expand_as(pos_trans_targets)
            mask_sum = max_trans_mask.float().sum() / 4
            
            #This will yield correct graph according to https://discuss.pytorch.org/t/how-to-use-condition-flow/644/5
            if mask_sum.data[0] > 0: 
#                print('WARNING: Masking out %d boxes in LocalizationLayer' % mask_sum.data[0])
                pos_trans[max_trans_mask] = 0
                pos_trans_targets[max_trans_mask] = 0
    
            # Compute RPN box regression loss
            weight = self.opt.mid_box_reg_weight
            reg_loss = weight * self.box_reg_loss.forward(pos_trans, pos_trans_targets)
        
        self.stats.losses.box_reg_loss = reg_loss.data
  
        # Fish out the box regression loss
        self.stats.losses.box_decay_loss = act_reg.data
  
        # Compute total loss
        total_loss = obj_loss_pos + obj_loss_neg + reg_loss + act_reg
        self.stats.losses.total_loss = total_loss.data
        
        if self.dtp_train:
            dtp_sampler_out = self.box_sampler_helper.forward(((input[3],), (gt_boxes, gt_embeddings)))
            dtp_pos_data, dtp_pos_target_data, dtp_neg_data, dtp_y = dtp_sampler_out
            dtp_pos_boxes = dtp_pos_data[0]
            dtp_pos_target_boxes, dtp_pos_target_labels = dtp_pos_target_data
            dtp_neg_boxes = dtp_neg_data[0]
            pos_boxes = torch.cat((pos_boxes, dtp_pos_boxes), dim=0)
            neg_boxes = torch.cat((neg_boxes, dtp_neg_boxes), dim=0)
            y = torch.cat((y, dtp_y), dim=0)
            pos_target_boxes = torch.cat((pos_target_boxes, dtp_pos_target_boxes), dim=0)
            pos_target_labels = torch.cat((pos_target_labels, dtp_pos_target_labels), dim=0)
            
        # Concatentate pos_boxes and neg_boxes into roi_boxes
        roi_boxes = torch.cat((pos_boxes, neg_boxes), dim=0)

        # Run the RoI pooling forward for roi_boxes
        self.roi_pooling.setImageSize(self.image_height, self.image_width)
        roi_features = self.roi_pooling.forward((cnn_features[0], roi_boxes))

        output = (roi_features, roi_boxes, pos_target_boxes, pos_target_labels, y, total_loss)
        return output
    
    #Clamp parallel arrays only to valid boxes (not oob of the image)
    def clamp_data(self, data, valid):
        #data should be 1 x kHW x D
        #valid is byte of shape kHW
        assert data.size(0) == 1, 'must have 1 image per batch'
        assert data.dim() == 3
        mask = valid.view(1, -1, 1).expand_as(data)
        return data[mask].view(1, -1, data.size(2))
        
    def _forward_test(self, input):
        cnn_features = input
        arg = easydict.EasyDict({'clip_boxes':self.test_clip_boxes,
               'nms_thresh':self.test_nms_thresh,
               'max_proposals':self.test_max_proposals})

        # Make sure that setImageSize has been called
        assert self.image_height and self.image_width and not self._called_forward_size, \
         'Must call setImageSize before each forward pass'
        self._called_forward_size = True
  
        rpn_out, act_reg = self.rpn.forward(cnn_features)
        rpn_boxes, rpn_anchors, rpn_trans, rpn_scores = rpn_out
        num_boxes = rpn_boxes.size(1)

        # Maybe clip boxes to image boundary
        if arg.clip_boxes:
            bounds = {'x_min':1, 'y_min':1,
                      'x_max':self.image_width,
                      'y_max':self.image_height}
            rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'xcycwh')

            #print(string.format('%d/%d boxes are predicted valid',
            #      torch.sum(valid), valid:nElement()))

            #Clamp parallel arrays only to valid boxes (not oob of the image)
            rpn_boxes = self.clamp_data(rpn_boxes, valid)
            rpn_anchors = self.clamp_data(rpn_anchors, valid)
            rpn_trans = self.clamp_data(rpn_trans, valid)
            rpn_scores = self.clamp_data(rpn_scores, valid)
            num_boxes = rpn_boxes.size(1)
  
        # Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
        rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes[0])
    
        # Convert objectness positive / negative scores to probabilities
        rpn_scores_exp = torch.exp(rpn_scores)
        pos_exp = rpn_scores_exp[0, :, 0]
        neg_exp = rpn_scores_exp[0, :, 1]
        scores = (pos_exp + neg_exp).pow(-1) * pos_exp
        
        verbose = False
        if verbose:
            print('in LocalizationLayer forward_test')
            print('Before NMS there are %d boxes'% num_boxes)
            print('Using NMS threshold %f' % arg.nms_thresh)
    
        #Run NMS and sort by objectness score
        boxes_scores = torch.cat((rpn_boxes_x1y1x2y2, scores.view(-1, 1)), dim=1)
      
        if arg.max_proposals == -1:
            idx = box_utils.nms(boxes_scores.data, arg.nms_thresh)
        else:
            idx = box_utils.nms(boxes_scores.data, arg.nms_thresh, arg.max_proposals)
            
        rpn_boxes_nms = torch.squeeze(rpn_boxes)[idx]
    
        if verbose:
            print('After NMS there are %d boxes' % rpn_boxes_nms.size(0))
    
        output = rpn_boxes_nms
        return output
  
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