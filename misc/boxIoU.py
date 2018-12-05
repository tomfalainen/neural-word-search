import torch
import numpy as np
import box_utils

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    
    from https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/utils/bbox.py
    """
    
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)

def boxIoU(boxes1, boxes2):
    boxes = box_utils.xcycwh_to_x1y1x2y2(boxes1)
    query_boxes = box_utils.xcycwh_to_x1y1x2y2(boxes2)
    return bbox_overlaps(boxes, query_boxes)

class BoxIoU(torch.nn.Module):
    def forward(self, input):
        box1 = input[0]
        box2 = input[1]
        N, B1, B2 = box1.size(0), box1.size(1), box2.size(1)
        area1 = box1[:, :, 2] * box1[:, :, 3]
        area2 = box2[:, :, 2] * box2[:, :, 3]
        area1_expand = area1.view(N, B1, 1).expand(N, B1, B2)
        area2_expand = area2.view(N, 1, B2).expand(N, B1, B2)
  
        box1_lohi = box_utils.xcycwh_to_x1y1x2y2_batch(box1) # N x B1 x 4
        box2_lohi = box_utils.xcycwh_to_x1y1x2y2_batch(box2) # N x B2 x 4
        box1_lohi_expand = box1_lohi.view(N, B1, 1, 4).expand(N, B1, B2, 4)
        box2_lohi_expand = box2_lohi.view(N, 1, B2, 4).expand(N, B1, B2, 4)
  
        x0 = torch.max(box1_lohi_expand[:, :, :, 0], box2_lohi_expand[:, :, :, 0])
        y0 = torch.max(box1_lohi_expand[:, :, :, 1], box2_lohi_expand[:, :, :, 1])
        x1 = torch.min(box1_lohi_expand[:, :, :, 2], box2_lohi_expand[:, :, :, 2])
        y1 = torch.min(box1_lohi_expand[:, :, :, 3], box2_lohi_expand[:, :, :, 3])
    
        w = (x1 - x0).clamp(min=0)
        h = (y1 - y0).clamp(min=0)
  
        intersection = w * h
        output = intersection * torch.pow(area1_expand + area2_expand - intersection, -1)
        return output
    
def overlaps(box1, box2):
    B1, B2 = box1.size(0), box2.size(0)
    area1 = box1[:, 2] * box1[:, 3]
    area2 = box2[:, 2] * box2[:, 3]
    area1_expand = area1.view(B1, 1).expand(B1, B2)
    area2_expand = area2.view(1, B2).expand(B1, B2)
  
    box1_lohi = box_utils.xcycwh_to_x1y1x2y2(box1) 
    box2_lohi = box_utils.xcycwh_to_x1y1x2y2(box2) 
    box1_lohi_expand = box1_lohi.view(B1, 1, 4).expand(B1, B2, 4)
    box2_lohi_expand = box2_lohi.view(1, B2, 4).expand(B1, B2, 4)
  
    x0 = torch.max(box1_lohi_expand[:, :, 0], box2_lohi_expand[:, :, 0])
    y0 = torch.max(box1_lohi_expand[:, :, 1], box2_lohi_expand[:, :, 1])
    x1 = torch.min(box1_lohi_expand[:, :, 2], box2_lohi_expand[:, :, 2])
    y1 = torch.min(box1_lohi_expand[:, :, 3], box2_lohi_expand[:, :, 3])

    w = (x1 - x0).clamp(min=0)
    h = (y1 - y0).clamp(min=0)
  
    intersection = w * h
    output = intersection * torch.pow(area1_expand + area2_expand - intersection, -1)
    return output
