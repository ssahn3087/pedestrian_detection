import torch
from torch.autograd import Variable


def euclidean_distance_loss(super, triplet_features):
    """
    euclidean_distance_loss
    """
    from torch.nn.functional import normalize
    anchor = normalize(triplet_features[0].view(-1), dim=0)
    positive = normalize(triplet_features[1].view(-1), dim=0)
    negative = normalize(triplet_features[2].view(-1), dim=0)

    pos_dist = ((anchor - positive) ** 2).sum(0)
    neg_dist = ((anchor - negative) ** 2).sum(0)

    if triplet_features.size(0) > 3:
        rem_features = normalize(triplet_features[3:], dim=1)
        bg_size = rem_features.size(0)
        rem_features = rem_features.view(bg_size, -1)
        rem_dist = ((anchor - rem_features) ** 2).sum(1).sum(0) / float(bg_size)
    else:
        rem_dist = 0.
    if super.debug:
        super.set += 1
        super.match += 1 if pos_dist.data[0] < neg_dist.data[0] and pos_dist.data[0] < rem_dist.data[0] else 0
    loss = pos_dist + (1.5 - 0.2*(neg_dist + rem_dist)).clamp(min=0.)
    return loss


def cross_entropy_l2_dist(super, triplet_features):
    """
    Binary Cross Entropy with l2 distance measurement
    """
    from torch.nn.functional import normalize
    from math import isnan
    match = True
    triplet_features = super.relu(triplet_features)
    triplet_features = normalize(triplet_features, dim=1)
    anchor = triplet_features[0].view(-1)
    positive = triplet_features[1].view(-1)
    negative = triplet_features[2].view(-1)

    scores = Variable(torch.zeros(3).cuda())
    scores[0] = 0.5 * torch.sqrt(((anchor - positive) ** 2).sum(0))
    scores[1] = 0.5 * torch.sqrt(((anchor - negative) ** 2).sum(0))
    labels = Variable(torch.ones(3).cuda())
    labels[0] = 0.
    match *= scores[0].data[0] < scores[1].data[0]

    if triplet_features.size(0) > 3:
        rem_features = normalize(triplet_features[3:], dim=1)
        bg_size = rem_features.size(0)
        rem_features = rem_features.view(bg_size, -1)
        scores[2] = 0.5 * torch.sqrt(((anchor - rem_features) ** 2).sum(1)).sum(0) / float(bg_size)
        match *= scores[0].data[0] < scores[2].data[0]
    else:
        scores = scores[:2]
        labels = labels[:2]
    if super.debug:
        super.set += 1
        super.match += 1 if match else 0
    loss = super.BCELoss(scores, labels) / scores.numel()
    return loss


def cross_entropy_cosine_sim(super, triplet_features):
    """
    Binary Cross Entropy (sigmoid included) with cosine similarity measurement
    """
    from torch.nn.functional import normalize, cosine_similarity
    from math import isnan
    match = True
    triplet_features = super.relu(triplet_features)
    triplet_features = normalize(triplet_features, dim=1)
    anchor = triplet_features[0].view(-1)
    positive = triplet_features[1].view(-1)
    negative = triplet_features[2].view(-1)

    scores = Variable(torch.zeros(3).cuda())
    scores[0] = cosine_similarity(anchor, positive, dim=0)
    scores[1] = cosine_similarity(anchor, negative, dim=0)
    labels = Variable(torch.zeros(3).cuda())
    labels[0] = 1.0
    match *= scores[0].data[0] > scores[1].data[0]

    if triplet_features.size(0) > 3:
        rem_features = normalize(triplet_features[3:], dim=1)
        bg_size = rem_features.size(0)
        anchor = anchor.unsqueeze(0)
        rem_features = rem_features.view(bg_size, -1)
        scores[2] = cosine_similarity(rem_features, anchor, dim=1).sum(0) / float(bg_size)
        match *= scores[0].data[0] > scores[2].data[0]
    else:
        scores = scores[:2]
        labels = labels[:2]
    if super.debug:
        super.set += 1
        super.match += 1 if match else 0
    loss = super.BCELoss(scores, labels) / scores.numel()
    return loss