import torch
import torch.nn.functional as F


def dice_loss(inputs, targets, num_objects):
    """
    Compute the DICE loss, similar to generalized IOU for masks.
    
    Args:
        inputs: A float tensor of shape [N, H, W]. Logits for each mask.
        targets: A float tensor of shape [N, H, W]. Binary ground truth.
        num_objects: Number of objects in the batch.
    
    Returns:
        Dice loss tensor (scalar).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    return loss.sum() / num_objects


def sigmoid_focal_loss(inputs, targets, num_objects, alpha: float = 0.25, gamma: float = 2):
    """
    Focal loss from RetinaNet: https://arxiv.org/abs/1708.02002.
    
    Args:
        inputs: A float tensor of shape [N, H, W]. Logits for each mask.
        targets: A float tensor of shape [N, H, W]. Binary ground truth.
        num_objects: Number of objects in the batch.
        alpha: Weighting factor for positive vs negative examples.
        gamma: Exponent of the modulating factor.
    
    Returns:
        Focal loss tensor (scalar).
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_objects


def iou_loss(inputs, targets, pred_ious, num_objects, use_l1_loss=False):
    """
    IoU prediction loss - compare predicted IoU scores to actual IoU.
    
    Args:
        inputs: A float tensor of shape [N, H, W]. Logits for each mask.
        targets: A float tensor of shape [N, H, W]. Binary ground truth.
        pred_ious: A float tensor of shape [N]. Predicted IoU scores.
        num_objects: Number of objects in the batch.
        use_l1_loss: Use L1 loss instead of MSE.
    
    Returns:
        IoU loss tensor (scalar).
    """
    # Compute actual IoU
    pred_mask = (inputs > 0).flatten(1)
    gt_mask = (targets > 0).flatten(1)
    
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    
    # Compare predicted vs actual IoU
    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    
    return loss.sum() / num_objects


def sam2_loss(
    pred_masks,
    gt_masks,
    pred_ious,
    num_objects,
    weight_dict,
    focal_alpha=0.25,
    focal_gamma=2.0,
    iou_use_l1=False,
):
    """
    Combined SAM2 loss: focal + dice + iou.
    
    Args:
        pred_masks: Predicted mask logits [N, H, W]
        gt_masks: Ground truth masks [N, H, W]
        pred_ious: Predicted IoU scores [N]
        num_objects: Number of objects
        weight_dict: Dictionary with keys 'loss_mask', 'loss_dice', 'loss_iou'
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        iou_use_l1: Use L1 for IoU loss
    
    Returns:
        dict with individual losses and total loss
    """
    
    assert 'loss_mask' in weight_dict, "weight_dict must contain 'loss_mask'"
    assert 'loss_dice' in weight_dict, "weight_dict must contain 'loss_dice'"
    assert 'loss_iou' in weight_dict, "weight_dict must contain 'loss_iou'"
    
    # Compute individual losses
    loss_mask = sigmoid_focal_loss(
        pred_masks, gt_masks, num_objects, alpha=focal_alpha, gamma=focal_gamma
    )
    
    loss_dice = dice_loss(pred_masks, gt_masks, num_objects)
    
    loss_iou = iou_loss(
        pred_masks, gt_masks, pred_ious, num_objects, use_l1_loss=iou_use_l1
    )
    
    total_loss = (
        weight_dict['loss_mask'] * loss_mask +
        weight_dict['loss_dice'] * loss_dice +
        weight_dict['loss_iou'] * loss_iou
    )
    
    return {
        'loss': total_loss,
        'loss_mask': loss_mask,
        'loss_dice': loss_dice,
        'loss_iou': loss_iou,
    }