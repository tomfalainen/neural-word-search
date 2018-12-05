import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import easydict

from misc.localization_layer import LocalizationLayer
from misc.box_regression_criterion import BoxRegressionCriterion
from misc.apply_box_transform import ApplyBoxTransform
from misc.logistic_loss import LogisticLoss

import misc.box_utils as box_utils
import misc.utils as utils
from misc.resnet_blocks import BasicBlock, Bottleneck

#to streamline code a bit
class myMultiLabelSoftMarginLoss(nn.Module):
    def __init__(self):
        super(myMultiLabelSoftMarginLoss, self).__init__()
        self.loss = nn.MultiLabelSoftMarginLoss()
    
    def forward(self, x, y, dummy):
        return self.loss(x, y)

class EmbeddingNet(nn.Module):
    def __init__(self, opt):
        super(EmbeddingNet, self).__init__()
        self.n_hidden = opt.n_hidden
        self.embedding_loss = opt.embedding_loss
        l = []
        l.append(self.get_seq(opt.ni, opt.nh))
        l += [self.get_seq(opt.nh, opt.nh) for i in range(self.n_hidden)]
        l.append(nn.Linear(opt.nh, opt.embedding_dim))
        self.net = nn.ModuleList(l)
        
    def get_seq(self, indim, outdim):
        a = []
        a.append(nn.Linear(indim, outdim))
        a.append(nn.BatchNorm1d(outdim))
        a.append(nn.Tanh())
        return nn.Sequential(*a)
        
    def init_weights(self, std):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0, std)
                module.bias.data.fill_(0.1)
        
    def forward(self, x):
        for m in self.net:
            x = m(x)
            
        if self.embedding_loss != 'phocnet':
            x = torch.div(x, torch.norm(x, 2, dim=1).view(-1, 1).expand_as(x))
        return x

class CtrlFNet(torch.nn.Module):
    def __init__(self, opt):
        super(CtrlFNet, self).__init__()
        utils.ensureopt(opt, 'mid_box_reg_weight')
        utils.ensureopt(opt, 'mid_objectness_weight')
        utils.ensureopt(opt, 'end_box_reg_weight')
        utils.ensureopt(opt, 'end_objectness_weight')
        utils.ensureopt(opt, 'embedding_weight')
        utils.ensureopt(opt, 'box_reg_decay')
        
        self.opt = opt
        self.emb2desc = {'dct':108, 'phoc':540}
        self.embedding_dim = self.emb2desc[self.opt.embedding]

        #output from bilinear interpolation, ensures that the output from layer4 is 2 x 5
        #TODO: infer one from the other, and also investigate different sizes?
        output_size = (8, 20) 
        if opt.num_layers == 34:
            input_dim = 128
        elif opt.num_layers == 50:
            input_dim = 512
            
        self.opt.output_size = output_size
        self.opt.input_dim = input_dim
        self.opt.cnn_dim = 512
        self.opt.contrastive_loss = self.opt.embedding_loss == 'cosine_embedding'
        
        x0, y0 = 0.0, 0.0
        sx, sy = 1.0, 1.0
        n = 4
            
        for i in range(n):
            x0 = x0 + sx / 2
            y0 = y0 + sy / 2
            sx = 2 * sx
            sy = 2 * sy
            
        self.opt.field_centers = (x0, y0, sx, sy)
        
        #First part of resnet
        block, layers = self.get_block_and_layers(opt.num_layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        #Localization layer
        self.localization_layer = LocalizationLayer(self.opt)
        
        #Rest of resnet
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AvgPool2d((2, 5))
        self.fc = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.fc.bias.data.zero_()
        
         #Initialize resnet weights
        self.init_weights()
        
        #Initialize localization_layer weights
        if opt.init_weights:
            self.localization_layer.init_weights()
        
        #Final box scoring layer
        self.box_scoring_branch = nn.Linear(512 * block.expansion, 2)
#        else:
#            self.box_scoring_branch = nn.Linear(512 * block.expansion, 1)
        if opt.init_weights:
            self.box_scoring_branch.weight.data.normal_(0, self.opt.std)
            self.box_scoring_branch.bias.data.zero_()
        
        #Final box regression layer
        self.apply_box_transform = ApplyBoxTransform()
        self.box_reg_branch = nn.Linear(512 * block.expansion, 4)
        self.box_reg_branch.weight.data.zero_()
        self.box_reg_branch.bias.data.zero_()
        
        #Embedding Net
        self.emb_opt = easydict.EasyDict({'ni':512 * block.expansion, 
                                          'nh':self.opt.emb_fc_size, 
                                          'embedding_dim':self.embedding_dim,
                                          'n_hidden': self.opt.embedding_net_layers, 
                                          'embedding_loss':self.opt.embedding_loss})
        self.embedding_net = EmbeddingNet(self.emb_opt)
            
        if opt.init_weights:
            self.embedding_net.init_weights(self.opt.std)
        
        #Losses
#        if self.opt.end_ce:
        self.scoring_loss = nn.CrossEntropyLoss()
#        else:
#            self.scoring_loss = LogisticLoss()

        self.box_reg_loss = BoxRegressionCriterion(self.opt.end_box_reg_weight)
        if self.opt.embedding_loss == 'cosine':
            self.embedding_loss = nn.CosineEmbeddingLoss(self.opt.cosine_margin)
        elif self.opt.embedding_loss == 'cosine_embedding':
            self.embedding_loss = nn.CosineEmbeddingLoss(self.opt.cosine_margin)
        elif self.opt.embedding_loss == 'BCE':
            self.embedding_loss = myMultiLabelSoftMarginLoss()
            
    def load_weights(self, weight_file):
        if weight_file:
            self.load_state_dict(torch.load(weight_file))
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_block_and_layers(self, num_layers):
        if num_layers == 34:
            out = BasicBlock, [3, 4, 6, 3]
        elif num_layers == 46:
            out = BasicBlock, [3, 4, 12, 3]
        elif num_layers == 50:    
            out = Bottleneck, [3, 4, 6, 3]
        else:
            raise ValueError, "invalid num_layer option"
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
      
        """ 
        Input: Dictoinary with the following keys:
        rpn_nms_thresh: NMS threshold for region proposals in the RPN; default is 0.7.
        final_nms_thresh: NMS threshold for final predictions; default is 0.3.
        num_proposals: Number of proposals to use; default is 1000
        """
    def setTestArgs(self, kwargs):
        self.localization_layer.setTestArgs({
              'nms_thresh':utils.getopt(kwargs, 'rpn_nms_thresh', 0.7),
        'max_proposals':utils.getopt(kwargs, 'max_proposals', 1000),})
        self.opt.final_nms_thresh = utils.getopt(kwargs, 'final_nms_thresh', 0.3)

    def setImageSize(self, image_height, image_width):
        self.localization_layer.setImageSize(image_height, image_width)
        
    def num_parameters(self):
        a = 0
        for p in self.parameters():
            a += p.data.nelement()
            
        return a
    
    def finetune_embedding(self):
        for p in self.parameters():
            p.requires_grad = False
            
        for p in self.embedding_net.parameters():
            p.requires_grad = True
            
    def _eval_helper(self, image, boxes, final_adjust_boxes):
        """
        Feeds boxes through the network in batches so that we aren't limited
        by the GPU memory when it comes to number of boxes at test time.
        """
        embed, scores, tboxes  = [], [], []
        for v in boxes.split(self.opt.test_batch_size):
            roi_feats = self.localization_layer.eval_boxes((image, Variable(v.cuda(), volatile=True)))
            roi_feats = self.layer3(roi_feats)
            roi_feats = self.layer4(roi_feats)
            roi_feats = self.bn2(roi_feats)
            roi_feats = self.relu(roi_feats)
            roi_feats = self.avgpool(roi_feats)
            roi_feats = roi_feats.view(roi_feats.size(0), -1)
            roi_codes = self.fc(roi_feats)
            s = self.box_scoring_branch(roi_codes).cpu()
            e = self.embedding_net(roi_codes).cpu()
            if final_adjust_boxes:
                box_trans = self.box_reg_branch(roi_codes)
                b = self.apply_box_transform((v, box_trans.data)).cpu()
                tboxes.append(b)
            
            embed.append(e.data)
            scores.append(s.data)
        
        embed = torch.cat(embed, dim=0)
        scores = torch.cat(scores, dim=0)
        out = (scores, embed)
        if final_adjust_boxes:
            tboxes = torch.cat(tboxes, dim=0)
            out += (tboxes,)
                
        return out
    
    #Clamp parallel arrays only to valid boxes (not oob of the image)
    def clamp_data(self, data, valid):
        #data should be kHW x D
        #valid is byte of shape kHW
        assert data.dim() == 2
        mask = valid.view(-1, 1).expand_as(data)
        return data[mask].view(-1, data.size(1))

    def evaluate(self, input, gpu, numpy=True, cpu=True):
        image, gt_boxes, proposals = input
        if gpu:
            image = image.cuda()
            
        B, C, H, W = image.shape
        self.setImageSize(H, W)

        image = Variable(image, volatile=True)
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)
        image = self.layer1(image)
        image = self.layer2(image)
        roi_boxes = self.localization_layer(image)
        gt_scores, gt_embed = self._eval_helper(image, gt_boxes, False)
        roi_scores, roi_embed, roi_boxes = self._eval_helper(image, roi_boxes.data, True)
        proposal_scores, proposal_embed = self._eval_helper(image, proposals, False)
        
        #Convert to x1y1x2y2
        roi_boxes = box_utils.xcycwh_to_x1y1x2y2(roi_boxes)

        if cpu:
            roi_scores = roi_scores.cpu()
            proposal_scores = proposal_scores.cpu()
            roi_boxes = roi_boxes.cpu()
            roi_embed = roi_embed.cpu()
            gt_embed = gt_embed.cpu()
            proposal_embed = proposal_embed.cpu()
        
        if numpy:
        #Convert to numpy array
            roi_scores = roi_scores.cpu().numpy()
            proposal_scores = proposal_scores.cpu().numpy()
            roi_boxes = roi_boxes.cpu().numpy()
            roi_embed = roi_embed.cpu().numpy()
            gt_embed = gt_embed.cpu().numpy()
            proposal_embed = proposal_embed.cpu().numpy()
            
        roi_scores = roi_scores[:, 1]
        proposal_scores = proposal_scores[:, 1]
            
        out = (roi_scores, proposal_scores, roi_boxes, roi_embed, gt_embed, proposal_embed)
        return out
        
    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            raise NotImplementedError("Don't call forward with model in eval mode, use evaluate instead")
        
    def _forward_train(self, input):
        image, gt_boxes, gt_embedding = input[0], input[1], input[2]
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)
        image = self.layer1(image)
        image = self.layer2(image)
        
        ll_in = (image, gt_boxes, gt_embedding)
        if self.opt.dtp_train:
            ll_in += (input[3], )
        
        roi_feats, roi_boxes, pos_target_boxes, pos_target_embeddings, y, mid_loss =  \
            self.localization_layer(ll_in)
        roi_feats = self.layer3(roi_feats)
            
        roi_feats = self.layer4(roi_feats)
        roi_feats = self.bn2(roi_feats)
        roi_feats = self.relu(roi_feats)
        roi_feats = self.avgpool(roi_feats)
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        roi_codes = self.fc(roi_feats)
        
        scores = self.box_scoring_branch(roi_codes)
        num_pos = pos_target_embeddings.size(0)
        pos_roi_codes = roi_codes[:num_pos]
        pos_roi_boxes = roi_boxes[:num_pos]
        
        #Don't do box regression on roi boxes since we don't try to adjust dtp boxes
        if self.opt.dtp_train: 
            reg_roi_codes = pos_roi_codes[:num_pos/2]
            reg_roi_boxes = pos_roi_boxes[:num_pos/2]
            box_trans = self.box_reg_branch(reg_roi_codes)
            boxes = self.apply_box_transform((reg_roi_boxes, box_trans))
        else:
            box_trans = self.box_reg_branch(pos_roi_codes)
            boxes = self.apply_box_transform((pos_roi_boxes, box_trans))
            
        embed = self.embedding_net(pos_roi_codes)
        return (scores, pos_roi_boxes, box_trans, boxes, embed, pos_target_boxes, 
                pos_target_embeddings, y, mid_loss)
    
    def forward_backward(self, data, gpu):
        self.train()
        
        img, gt_boxes, gt_embeddings = data[0], data[1], data[2]
        
        if gpu:
            img = img.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_embeddings = gt_embeddings.cuda()
            
        input = (Variable(img), Variable(gt_boxes.float()), Variable(gt_embeddings.float()))
        self.setImageSize(img.size(2), img.size(3))
        
        if self.opt.dtp_train:
            dtp = data[4]
            if gpu:
                dtp = dtp.cuda()
            
            input += (Variable(dtp.float()), )
        
        out = self.forward(input)
        wordness_scores = out[0]
        pos_roi_boxes = out[1]
        final_box_trans = out[2]
        emb_output = out[4]
        gt_boxes = out[5]
        gt_embeddings = out[6]
        y = out[7]
        mid_loss = out[8]

        num_boxes = wordness_scores.size(0)
        num_pos = pos_roi_boxes.size(0)

        # Compute final objectness loss and gradient
        wordness_labels = torch.zeros(num_boxes).view(-1, 1)
        wordness_labels[:num_pos].fill_(1)      
        wordness_labels = Variable(wordness_labels)
        if gpu:
            wordness_labels = wordness_labels.cuda()
            
        wordness_labels = wordness_labels.long().view(-1)
            
        end_objectness_loss = self.scoring_loss.forward(wordness_scores, wordness_labels) \
                                    * self.opt.end_objectness_weight

        if self.opt.dtp_train:
            pos_roi_boxes = pos_roi_boxes[:num_pos / 2]
            gt_boxes = gt_boxes[:num_pos / 2]
            
        # this one multiplies by the weight inside the loss so we don't do it manually.
        end_box_reg_loss = self.box_reg_loss.forward((pos_roi_boxes, final_box_trans), gt_boxes)

        embedding_loss = self.embedding_loss.forward(emb_output, gt_embeddings, y) \
                            * self.opt.embedding_weight
    
        total_loss = mid_loss + end_objectness_loss + end_box_reg_loss + embedding_loss
        total_loss.backward()
        
        ll_losses = self.localization_layer.stats.losses
        losses = {
        'mo':ll_losses.obj_loss_pos.cpu() + ll_losses.obj_loss_neg.cpu(),
        'bd':ll_losses.box_decay_loss.cpu(),
        'mbr':ll_losses.box_reg_loss.cpu(),
        'eo':end_objectness_loss.data.cpu(),
        'ebr':end_box_reg_loss.data.cpu(),
        'e':embedding_loss.data.cpu(),
        'total_loss':total_loss.data.cpu(),
        }
        
        for k, v in losses.iteritems():
            losses[k] = v.item()
        
        return losses
