import torch
import torch.nn.functional as F

from .utils import *

"""
Todo: implement different losses
1. cosine sim (done)
2. 2 - 2 * cosine sim 
3. L2 loss (done)
4. L1 loss (done)

Compare
1. v9, cosine, l1, l2
2. cosine softmax, cosine max all
"""

def L1_distance_max_all(sp_learn, h1, t1, h0, t0, context):
    """
    Use L1 distance to measure (dis)similarity between
    t0's.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        dist = []
        bs = t0.size(0)
        for i in range(bs):
            d = (t0[i:i+1] - t0).abs().sum(1).view(1, -1)
            dist.append(d)
        dist = torch.cat(dist)

    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    # this part stays the same
    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yym = dist.min(1,keepdim=True)[0]
    yy = (yy == yym).float()

    l = soft_target_cross_entropy(y, yy)

    return l

def L1_distance_softmax(sp_learn, h1, t1, h0, t0, context):
    """
    Use L1 distance to measure (dis)similarity between
    t0's.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        dist = []
        bs = t0.size(0)
        for i in range(bs):
            d = (t0[i:i+1] - t0).abs().sum(1).view(1, -1)
            dist.append(d)
        dist = torch.cat(dist)

    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    # this part stays the same
    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yy = torch.softmax(-dist, dim=-1)

    l = soft_target_cross_entropy(y, yy)

    return l

def L2_distance_max_all(sp_learn, h1, t1, h0, t0, context):
    """
    Use L2 distance to measure (dis)similarity between
    t0's.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        dist = []
        bs = t0.size(0)
        for i in range(bs):
            d = (t0[i:i+1] - t0).square().sum(1).view(1, -1)
            dist.append(d)
        dist = torch.cat(dist).sqrt()

    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    # this part stays the same
    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yym = dist.min(1,keepdim=True)[0]
    yy = (yy == yym).float()

    l = soft_target_cross_entropy(y, yy)

    return l

def L2_distance_softmax(sp_learn, h1, t1, h0, t0, context):
    """
    Use L2 distance to measure (dis)similarity between
    t0's.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        dist = []
        bs = t0.size(0)
        for i in range(bs):
            d = (t0[i:i+1] - t0).square().sum(1).view(1, -1)
            dist.append(d)
        dist = torch.cat(dist).sqrt()

    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    # this part stays the same
    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yy = torch.softmax(-dist, dim=-1)

    l = soft_target_cross_entropy(y, yy)

    return l

def cosine_sim_softmax(sp_learn, h1, t1, h0, t0, context):
    """
    Similar as v9_input_target_max_all
    Use cosine similarity as the similarity metric
    instead of dot product.
    Implemented by normalizing t0 simply.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        # normalize t0
        t0 = t0 / t0.sum(1, keepdims=True)
        yy = t0 @ t0.t()
    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yy = torch.softmax(yy, dim=-1)

    l = soft_target_cross_entropy(y, yy)

    return l

def cosine_sim_max_all(sp_learn, h1, t1, h0, t0, context):
    """
    Similar as v9_input_target_max_all
    Use cosine similarity as the similarity metric
    instead of dot product.
    Implemented by normalizing t0 simply.
    """
    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        # normalize t0
        t0 = t0 / t0.sum(1, keepdims=True)
        yy = t0 @ t0.t()
    else:
        raise NotImplementedError("Not implemented for images")
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yym = yy.max(1,keepdim=True)[0]
    yy = (yy == yym).float()

    l = soft_target_cross_entropy(y, yy)

    return l

def v9_input_target_max_all(sp_learn,h1,t1,h0,t0,context):

    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        yy = t0 @ t0.t()
    else:
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    yym = yy.max(1,keepdim=True)[0]
    yy = (yy == yym).float()

    l = soft_target_cross_entropy(y, yy)

    return l

def v14_input_target_max_rand(sp_learn,h1,t1,h0,t0,context):

    if len(t0.shape) == 2:
        t0 = t0.flatten(1)
        yy = t0 @ t0.t()
    else:
        t0 = t0.flatten(2).permute(2,0,1)
        yy = (t0 @ t0.permute(0,2,1)).mean(0)

    if len(t1.shape) > 2:
        h1 = h1.flatten(2).permute(2,0,1)
        t1 = t1.flatten(2).permute(2,1,0)
        y = (h1 @ t1).mean(0)

    else:
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)

        y = h1 @ t1.t()

    l  = F.cross_entropy(y, yy.argmax(1))

    return l

class v15_input_target_topk(object):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def __call__(self,sp_learn,h1,t1,h0,t0,context):

        if len(t0.shape) == 2:
            t0 = t0.flatten(1)
            yy = t0 @ t0.t()
        else:
            t0 = t0.flatten(2).permute(2,0,1)
            yy = (t0 @ t0.permute(0,2,1)).mean(0)

        if len(t1.shape) > 2:
            h1 = h1.flatten(2).permute(2,0,1)
            t1 = t1.flatten(2).permute(2,1,0)
            y = (h1 @ t1).mean(0)

        else:
            h1 = h1.flatten(1)
            t1 = t1.flatten(1)

            y = h1 @ t1.t()

        yym = yy.topk(self.topk,dim=1)[1]
        yy = F.one_hot(yym,yy.shape[1]).sum(1).clamp(0,1).float()

        l = soft_target_cross_entropy(y, yy)

        return l
