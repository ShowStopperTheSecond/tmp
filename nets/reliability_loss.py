# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch.nn as nn
import torch.nn.functional as F

from nets.ap_loss import APLoss
from nets.pnp_loss import  PNPLoss
import torch




class PixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow)
        
        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0
        scores, gt = scores.view(n,-1), gt.view(n,-1)
        ap = self.aploss(scores, gt).view(msk.shape)

        pixel_loss = self.loss_from_ap(ap, qconf)
        
        loss = pixel_loss[msk].mean()
        return loss




class ReliabilityLoss (PixelAPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """
    def __init__(self, sampler, base=0.5, **kw):
        PixelAPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'


    def loss_from_ap(self, ap, rel):
        return 1 - ap*rel - (1-rel)*self.base






class PixelPNPLoss(nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.

        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """

    def __init__(self, sampler,   b=2, alpha=1, anneal=0.01, variant="Dq",):
        nn.Module.__init__(self)
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        self.pnploss = PNPLoss(b, alpha,anneal,variant)
        self.name = 'pixPNP'
        self.sampler = sampler


    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow)

        # compute pixel-wise PNP Loss
        n = qconf.numel()
        if n == 0: return 0

        pnp = self.pnploss(scores, gt)['loss']['losses']
        pnp =  pnp.view(msk.shape)
        # pnp = nn.functional.normalize(pnp, p=2.0, dim=1, eps=1e-12, out=None)


        pixel_loss = self.loss_from_pnp(pnp, qconf)

        loss = pixel_loss[msk].mean()

        return loss


class ReliabilityPNPLoss(PixelPNPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """

    def __init__(self, sampler, base=0.5, **kw):
        PixelPNPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'

    def loss_from_pnp(self, pnp, rel):
        return 1 - pnp * rel - (1 - rel) * self.base





class CustomPixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow)
        
        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0
        scores, gt = scores.view(n,-1), gt.view(n,-1)
        ap = self.aploss(scores, gt).view(msk.shape)

        pixel_loss = self.loss_from_ap(ap, qconf)
        
        loss = pixel_loss[msk].mean()


        return loss




class CustomReliabilityLoss (CustomPixelAPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """
    def __init__(self, sampler, base=0.5, **kw):
        CustomPixelAPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'
        self.cosine_similarity =  nn.CosineSimilarity(dim=0, eps=1e-6)



    def loss_from_ap(self, ap, rel):

        return 2  -ap   - self.cosine_similarity(ap, rel) - (1-rel)*self.base

        # return torch.abs(1 - self.cosine_similarity(ap, rel) - (1-rel)*self.base)
        # return 1 - self.cosine_similarity(ap, rel) - (1-rel)*self.base

        # return 1 - ap*rel - (1-rel)*self.base






class PixelPNPLoss(nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.

        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """

    def __init__(self, sampler, b=2, alpha=1, anneal=0.01, variant="Dq", ):
        nn.Module.__init__(self)
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        self.pnploss = PNPLoss(b, alpha, anneal, variant)
        self.name = 'pixPNP'
        self.sampler = sampler

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf, anchor, positive_samples, negative_samples, distractor_samples = self.sampler(descriptors, kw.get('reliability'), aflow)

        # compute pixel-wise PNP Loss
        n = qconf.numel()
        if n == 0: return 0

        pnp = self.pnploss(scores, torch.arange(0, len(scores)), gt,(anchor, positive_samples, negative_samples, distractor_samples ) )['loss']['losses']
        pnp = pnp.view(msk.shape)
        # pnp = nn.functional.normalize(pnp, p=2.0, dim=1, eps=1e-12, out=None)

        pixel_loss = self.loss_from_pnp(pnp, qconf)

        loss = pixel_loss[msk].mean()

        return loss


class ReliabilityPNPLoss(PixelPNPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """

    def __init__(self, sampler, base=0.5, **kw):
        PixelPNPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'

    def loss_from_pnp(self, pnp, rel):
        return 1 - pnp * rel - (1 - rel) * self.base
