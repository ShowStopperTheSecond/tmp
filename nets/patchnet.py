# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F



class GrowingCosineUnit(torch.nn.Module):
    def __init__(self):
        super(GrowingCosineUnit, self).__init__()

    def forward(self, z):
        return z * torch.cos(z)



# class BaseNet (nn.Module):
#     """ Takes a list of images as input, and returns for each image:
#         - a pixelwise descriptor
#         - a pixelwise confidence
#     """
#     def softmax(self, ux):
#         if ux.shape[1] == 1:
#             x = F.softplus(ux)
#             return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
#         elif ux.shape[1] == 2:
#             return F.softmax(ux, dim=1)[:,1:2]

#     def normalize(self, x, ureliability, urepeatability):
#         if len(x) == 2:
#             ret_val = dict(descriptors = F.normalize(x, p=2, dim=1),
#                     repeatability = self.softmax( urepeatability ),
#                     reliability = self.softmax( ureliability ))
#         else:
#             normalized_xs = []
#             for feats in x:
#                 normalized_xs.append( F.normalize(feats, p=2, dim=1))
#             ret_val = dict(descriptors = normalized_xs,
#                     repeatability = self.softmax( urepeatability ),
#                     reliability = self.softmax( ureliability ))


#         return  ret_val

#     def forward_one(self, x):
#         raise NotImplementedError()

#     def forward(self, imgs, **kw):
#         res = [self.forward_one(img) for img in imgs]
#         # merge all dictionaries into one
#         res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
#         return dict(res, imgs=imgs, **kw)


class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def normalize2(self, x, ureliability, urepeatability):
     
        normalized_xs = []
        for feats in x:
            normalized_xs.append( F.normalize(feats, p=2, dim=1))
        ret_val = dict(descriptors = normalized_xs,
                repeatability = self.softmax( urepeatability ),
                reliability = self.softmax( ureliability ))
        return ret_val


    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)



class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, gcu=False, selu=False, mish=False, k_pool = 1, pool_type='max'):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation * dilation
        if self.dilated: 
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )
        if bn and self.bn: self.ops.append( self._make_bn(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        if gcu: self.ops.append(GrowingCosineUnit())
        if selu: self.ops.append(nn.SELU(inplace=False))
        if mish: self.ops.append(torch.nn.modules.activation.Mish(inplace=False))
        # if softsign: self.opt.append(torch.nn.modules.activation.Softsign())

        self.curchan = outd
        
        if k_pool > 1:
            if pool_type == 'avg':
                self.ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                self.ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")
    
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)


class L2_Net (PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.

        From the L2Net paper (CVPR'17).
    """
    def __init__(self, dim=128, **kw ):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n,**kw: self._add_conv((n*dim)//128,**kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim




class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, stride=2)
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim



class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)





class Custom_Quad_L2Net_GCU (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan, relu=False, gcu=True)
        self._add_conv(  8*mchan,relu=False, gcu=True)
        self._add_conv( 16*mchan, stride=2,relu=False, gcu=True)
        self._add_conv( 16*mchan,relu=False, gcu=True)
        self._add_conv( 32*mchan, stride=2,relu=False, gcu=True)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22, gcu=True)
        self._add_conv( 32*mchan, k=2, stride=2, relu=False, gcu=True)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim



class Custom_Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            if op._get_name() == "ReLU":
            # if op._get_name() == "GrowingCosineUnit":
                descriptors.append(x)
            x = op(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-4:], ureliability, urepeatability)



class Custom_2_Quad_L2Net_ConfCFS(Custom_Quad_L2Net_GCU):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Custom_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        # descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            # if op._get_name() == "GrowingCosineUnit":
            #     descriptors.append(x)
            x = op(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize(x, ureliability, urepeatability)



class Fast_Quad_L2Net (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, k_pool = downsample_factor) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        
        # Go back to initial image resolution with upsampling
        self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
        
        self.out_dim = dim
        

        
class Fast_Quad_L2Net_ConfCFS (Fast_Quad_L2Net):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Fast_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)



class Custom_3_Fast_Quad_L2Net (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=True, gcu=False, mish=False)
        self._add_conv(  8*mchan,relu=True, gcu=False, mish=False)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=True, gcu=False, mish=False) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan,relu=True, gcu=False, mish=False)
        self._add_conv( 32*mchan,relu=True, gcu=False, stride=2, mish=False)
        self._add_conv( 32*mchan,relu=True, gcu=False, mish=False)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=True, gcu=False, mish=False)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22, mish=False)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=True, gcu=False, mish=False)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_3_Fast_Quad_L2Net_ConfCFS (Custom_3_Fast_Quad_L2Net):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_3_Fast_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            if op._get_name() == "ReLU":
            # if op._get_name() == "Mish":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)




class Custom_3_Fast_Quad_L2Net_Mish (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=False, gcu=False, mish=True) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, mish=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, mish=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, mish=True)
        self._add_conv( 32*mchan, k=2, stride=2, relu=False, mish=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, mish=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_3_Fast_Quad_L2Net_ConfCFS_Mish (Custom_3_Fast_Quad_L2Net_Mish):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_3_Fast_Quad_L2Net_Mish.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "Mish":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)






class Custom_3_Fast_Quad_L2Net_Selu (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=False, gcu=False, mish=True) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, mish=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, mish=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, mish=True)
        self._add_conv( 32*mchan, k=2, stride=2, relu=False, mish=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, mish=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_3_Fast_Quad_L2Net_ConfCFS_Selu (Custom_3_Fast_Quad_L2Net_Selu):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_3_Fast_Quad_L2Net_Selu.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "SELU":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)









class Custom_4_Fast_Quad_L2Net_Selu (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=False, gcu=False, selu=True) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, selu=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, selu=True)
        # self._add_conv( 32*mchan, k=2, stride=2, relu=False, selu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, selu=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_4_Fast_Quad_L2Net_ConfCFS_Selu (Custom_4_Fast_Quad_L2Net_Selu):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_4_Fast_Quad_L2Net_Selu.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "SELU":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)





class Custom_5_Fast_Quad_L2Net_Selu (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=False, gcu=False, selu=True) # added avg pooling to decrease img resolution
        # self._add_conv( 16*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, selu=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, selu=True)
        # self._add_conv( 32*mchan, k=2, stride=2, relu=False, selu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, selu=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_5_Fast_Quad_L2Net_ConfCFS_Selu (Custom_5_Fast_Quad_L2Net_Selu):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_5_Fast_Quad_L2Net_Selu.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "SELU":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)




class Custom_6_Fast_Quad_L2Net_Mish (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,mish=False, gcu=False, selu=True) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, mish=True)
        # self._add_conv( 32*mchan,relu=False, gcu=False, mish=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, mish=True)
        # self._add_conv( 32*mchan, k=2, stride=2, relu=False, selu=relu22)
        self._add_conv( dim, k=2, stride=4, relu=False, mish=True)
        
        # self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, mish=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_6_Fast_Quad_L2Net_ConfCFS_Mish (Custom_6_Fast_Quad_L2Net_Mish):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_6_Fast_Quad_L2Net_Mish.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "Mish":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)





class Custom_5_Fast_Quad_L2Net_Mish (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv(  8*mchan,relu=False, gcu=False, mish=True)
        self._add_conv( 16*mchan, k_pool = downsample_factor,mish=False, gcu=False, selu=True) # added avg pooling to decrease img resolution
        # self._add_conv( 16*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, mish=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, mish=True)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, mish=True)
        # self._add_conv( 32*mchan, k=2, stride=2, relu=False, selu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, mish=True)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_5_Fast_Quad_L2Net_ConfCFS_Mish (Custom_5_Fast_Quad_L2Net_Mish):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_5_Fast_Quad_L2Net_Mish.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "Mish":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)







class Custom_6_Fast_Quad_L2Net_Selu (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self.downsample_factor = downsample_factor
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True, bn=False)
        self._add_conv(  8*mchan,relu=False, gcu=False, selu=True, bn=False)
        self._add_conv( 16*mchan, k_pool = downsample_factor,relu=False, gcu=False, selu=True,, bn=False) # added avg pooling to decrease img resolution
        # self._add_conv( 16*mchan,relu=False, gcu=False, selu=True)
        self._add_conv( 32*mchan,relu=False, gcu=False, stride=2, selu=True,, bn=False)
        self._add_conv( 32*mchan,relu=False, gcu=False, selu=True,, bn=False)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2,relu=False, gcu=False, selu=True, bn=False)
        # self._add_conv( 32*mchan, k=2, stride=2, relu=False, selu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False,relu=False, gcu=False, selu=True, bn=False)
        
        # Go back to initial image resolution with upsampling
        # self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
      
        self.out_dim = dim
        

        
class Custom_6_Fast_Quad_L2Net_ConfCFS_Selu (Custom_6_Fast_Quad_L2Net_Selu):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Custom_6_Fast_Quad_L2Net_Selu.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        descriptors = []
        for op in self.ops:
            # if op._get_name() == "ReLU":
            if op._get_name() == "SELU":
            # if op._get_name() == "GrowingCosineUnit":
               descriptors.append(
                torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x))
            x = op(x)
        x =  torch.nn.Upsample(scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize2(descriptors[-5:], ureliability, urepeatability)
