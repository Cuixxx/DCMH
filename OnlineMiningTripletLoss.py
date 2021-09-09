#reference  https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
import torch
import torch.nn.functional as F
import numpy as np
def pairwise_distance(embeddings,squared=False):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dot_product = torch.matmul(embeddings, embeddings.t())
    # square_norm = torch.diag(dot_product)
    # distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = 2.0 * (1-dot_product)
    distances[distances < 0] = 0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0-mask)*torch.sqrt(distances)

    return distances
"""
def pairwise_distance(embeddings,squared=False):
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0

    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0-mask)*torch.sqrt(distances)

    return distances

"""

def cm_pairwise_distance(embedding0,embedding1,squared=False):
    embedding0 = F.normalize(embedding0, p=2, dim=1)
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    dot_product = torch.matmul(embedding0, embedding1.t())
    # square0 = torch.diag(torch.matmul(embedding0, embedding0.t()))
    # square1 = torch.diag(torch.matmul(embedding1, embedding1.t()))
    distances = 2.0*(1 - dot_product)
    distances[distances < 0] = 0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0-mask)*torch.sqrt(distances)
    return distances


def get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0)).type(torch.uint8).cuda()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = ~i_equal_k & i_equal_j
    return distinct_indices&valid_labels

def cm_get_triplet_mask(labels):
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = ~i_equal_k & i_equal_j
    return valid_labels

def get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return indices_not_equal & labels_equal

def cm_get_anchor_positive_triplet_mask(labels):

    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
    # x = labels_equal.cpu().numpy()
    # print(x)
    return labels_equal

def get_anchor_negative_triplet_mask(labels):
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def cm_get_anchor_negative_triplet_mask(labels):
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, device='cpu'):
    pairwis_dist = pairwise_distance(embeddings,squared=squared)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels,device)
    anchor_positive_dist = mask_anchor_positive*pairwis_dist
    hardest_positve_dist,_ = anchor_positive_dist.max(dim=1, keepdim=True)
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
    max_anchor_negative_dist,_ = pairwis_dist.max(dim=1, keepdim=True)
    anchor_negative_dist = (1-mask_anchor_negative)*max_anchor_negative_dist+pairwis_dist
    hardest_negative_dist,_ = anchor_negative_dist.min(dim=1, keepdim=True)
    triplet_loss = hardest_positve_dist - hardest_negative_dist + margin
    triplet_loss[triplet_loss<0]=0
    triplet_loss = triplet_loss.mean()

    return triplet_loss

def cm_batch_hard_triplet_loss(labels, anchor, another, margin, squared=False):
    pairwis_dist = cm_pairwise_distance(anchor, another, squared=squared)
    mask_anchor_positive = cm_get_anchor_positive_triplet_mask(labels).float()
    # print(mask_anchor_positive)
    anchor_positive_dist = mask_anchor_positive * pairwis_dist

    ap_dist = anchor_positive_dist.sum()/mask_anchor_positive.sum()

    hardest_positve_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
    mask_anchor_negative = cm_get_anchor_negative_triplet_mask(labels).float()
    anchor_negative_dist = mask_anchor_negative * pairwis_dist
    an_dist = anchor_negative_dist.sum()/mask_anchor_negative.sum()
    max_anchor_negative_dist, _ = pairwis_dist.max(dim=1, keepdim=True)
    anchor_negative_dist = (1 - mask_anchor_negative) * max_anchor_negative_dist + pairwis_dist
    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
    triplet_loss = hardest_positve_dist - hardest_negative_dist + margin
    triplet_loss[triplet_loss < 0] = 0
    triplet_loss = triplet_loss.mean()

    return triplet_loss,ap_dist,an_dist


def batch_all_triplet_loss(labels, embedings, margin, squared=False):

    pairwise_dist = pairwise_distance(embedings, squared)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    triplet_loss[triplet_loss < 0] = 0

    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)
    triplet_loss = triplet_loss.sum()/(float(num_positive_triplets) + 1e-16)

    return triplet_loss, fraction_positive_triplets

def cm_batch_all_triplet_loss(labels, anchor, another, margin, squared=False):


    pairwise_dist = cm_pairwise_distance(anchor, another, squared)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)

    ap_mask = cm_get_anchor_positive_triplet_mask(labels).float()
    ap_dist = (pairwise_dist*ap_mask).sum()/ap_mask.sum()
    an_mask = cm_get_anchor_negative_triplet_mask(labels).float()
    an_dist = (pairwise_dist*an_mask).sum()/an_mask.sum()

    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = cm_get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    triplet_loss[triplet_loss < 0] = 0

    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (float(num_valid_triplets) + 1e-16)
    triplet_loss = triplet_loss.sum()/(float(num_positive_triplets) + 1e-16)

    return triplet_loss, fraction_positive_triplets, ap_dist, an_dist

