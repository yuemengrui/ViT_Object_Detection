# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        return torch.tensor([-1.0])
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        return torch.tensor([-1.0])
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class SetCriterion(nn.Module):

    def __init__(self, loss_coef=None):
        super().__init__()
        if loss_coef is None:
            self.loss_coef = {'bbox': 5, 'giou': 2}

    def loss_boxes(self, outputs, target_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        loss_bbox = F.l1_loss(outputs, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum()

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(outputs), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum()
        losses['loss'] = losses['loss_bbox'] * self.loss_coef['bbox'] + losses['loss_giou'] * self.loss_coef['giou']
        return losses

    def forward(self, outputs, targets):
        losses = self.loss_boxes(outputs, targets)

        return losses
