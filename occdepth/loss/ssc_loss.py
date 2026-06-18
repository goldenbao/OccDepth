import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


def FocalLoss(pred, target, class_weights, gamma=2.0):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred: (B, C, H, W, D) raw logits
        target: (B, H, W, D) long, ignore_index=255
        class_weights: (C,) per-class weights
        gamma: focusing parameter (default 2.0)
    """
    # Per-element CE loss (no reduction, no weighting yet)
    ce = F.cross_entropy(pred, target.long(), reduction="none", ignore_index=255)

    # Softmax probabilities of the target class (p_t)
    p = F.softmax(pred, dim=1)
    mask = target != 255
    p_t = p.gather(1, target.unsqueeze(1).clamp(min=0))  # clamp for ignore_index
    p_t = p_t.squeeze(1)  # (B, H, W, D)

    # Focal factor: (1 - p_t)^gamma
    focal_factor = (1.0 - p_t) ** gamma

    # Gather class weights for each target
    weight = class_weights[target.clamp(min=0)]  # (B, H, W, D)

    # Combine: weight * focal_factor * CE
    loss = (weight * focal_factor * ce).sum() / mask.float().sum().clamp(min=1.0)
    return loss


def DiceLoss(pred, target, class_weights, smooth=1.0):
    """
    Dice Loss: 1 - Dice coefficient, per-class weighted average.

    Args:
        pred: (B, C, H, W, D) raw logits
        target: (B, H, W, D) long, ignore_index=255
        class_weights: (C,) per-class weights
        smooth: smoothing factor (default 1.0)
    """
    n_classes = pred.shape[1]
    p = F.softmax(pred, dim=1)  # (B, C, H, W, D)

    # One-hot target
    target_clamped = target.clone()
    target_clamped[target == 255] = 0  # temporary, we mask later
    t = F.one_hot(target_clamped.long(), num_classes=n_classes)  # (B, H, W, D, C)
    t = t.permute(0, 4, 1, 2, 3).float()  # (B, C, H, W, D)

    # Mask: ignore unknown voxels in both pred and target
    mask = (target != 255).float()  # (B, H, W, D)
    mask = mask.unsqueeze(1)  # (B, 1, H, W, D)

    intersection = (p * t * mask).sum(dim=(0, 2, 3, 4))  # (C,)
    cardinality = ((p + t) * mask).sum(dim=(0, 2, 3, 4))  # (C,)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)  # (C,)

    # Weighted mean over classes with non-zero weight
    valid_mask = class_weights > 0
    dice_loss = 1.0 - dice  # (C,)
    loss = (dice_loss * class_weights).sum() / class_weights[valid_mask].sum()
    return loss
