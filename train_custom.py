import os, pdb
import torch
import torch.optim as optim

from tools import common, trainer
from tools.dataloader import *
from nets.patchnet import *
from nets.losses import *
import torchvision.transforms.functional as transform
from PIL import Image, ImageOps
from tqdm.notebook import tqdm


from  datasets import *

import matplotlib.pyplot as plt
import cv2



# default_net = "Quad_L2Net_ConfCFS()"
default_net = "Custom_Quad_L2Net_ConfCFS()"

toy_db_debug = """SyntheticPairDataset(
    ImgFolder('imgs'), 
            'RandomScale(256,1024,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(.5)')"""

db_web_images = """SyntheticPairDataset(
    web_images, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(.5)')"""

db_aachen_images = """SyntheticPairDataset(
    aachen_db_images, 
        'RandomScale(256,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(.5)')"""

db_aachen_style_transfer = """TransformedPairs(
    aachen_style_transfer_pairs,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(.5)')"""

db_aachen_flow = "aachen_flow_pairs"


db_sar_images = """SyntheticPairDataset(
    sar_db_images, 
        'RandomScale(256,256,can_upscale=False)', 
        'RandomTilting(0.5), PixelSpeckleNoise(.5)')"""

db_sar_flow = " sar_db_flow"



default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(256,1024,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(192)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""

# default_sampler = """DoubleDescNghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
#                             subd_neg=-8,maxpool_pos=True, desc_dim=128)"""
#


default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""


data_sources = dict(
    # D = toy_db_debug,
    # W = db_web_images,
    # A = db_aachen_images,
    # F = db_aachen_flow,
    # S = db_aachen_style_transfer,
    # X = db_sar_images,
    Z = db_sar_flow
    )


def fix_img(img):
    c = (img - img.min()) / (img.max() - img.min()) * 255
    return c



class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """
    def forward_backward(self, inputs):
        img1 =  np.array(inputs['img1'][0])
        img2 =  np.array(inputs['img2'][0])
        mask = np.array(inputs['mask'][0])
        aflow = np.array(inputs['aflow'][0])
        np.save('img1', img1)
        np.save('img2', img2)
        np.save('mask', mask )
        np.save('aflow', aflow)
        # np.save('transformation', final_tr_mat)
        images = [inputs.pop('img1'),inputs.pop('img2')]
        output = self.net(imgs=images)
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details


def load_network(model_fn):
    checkpoint = torch.load(model_fn, map_location=torch.device('cpu'))
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


def load_network_custom(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + str(checkpoint['net']))
    net = checkpoint['net']
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


# For evaluating
class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    catched = False
    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]
            if not catched:
                ret_reliability = reliability.cpu().detach().numpy().squeeze()
                ret_repeatability = repeatability.cpu().detach().numpy().squeeze()
                catched = True

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = torch.nn.functional.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores, ret_reliability, ret_repeatability

def to_opencv_keypoints(keypoints,size_scale=20):
    converterd_keypoints=[]
    for x, y, scale in keypoints:
        k=cv2.KeyPoint(x=x, y=y, size=scale, angle=0, octave=int(scale))
        converterd_keypoints.append(k)
    return converterd_keypoints
def translatePoint(p, d):
    x, y = p
    dx, dy = d
    return [x + dx, y + dy]

def homographyTransform(h,p):
    res=h@p
    res=res/res[-1]
    return res[:2].astype('int')



def drawMatches(pt1, img1, pt2, img2, mask, h_mat):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros(shape=(max(h1, h2), w1 + w2), dtype='uint8')
    img[:h1, :w1] = img1
    img[:h2, w1:] = img2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    x = []
    x.append([0, 0, 1])
    x.append([w1, 0, 1])
    x.append([w1, h1, 1])
    x.append([0, h1, 1])

    tx = [translatePoint(homographyTransform(h_mat, np.array(p)), (w1, 0)) for p in x]
    tx = np.array([tx])
    for i, m in enumerate(mask):
        if m == 1:
            dst = (int(w1 + pt2[i][0]), int(pt2[i][1]))
            src = (int(pt1[i][0]), int(pt1[i][1]))
            #             color=tuple(np.random.randint(0,255,3,dtype='int'))
            color = np.random.randint(0, 255, size=(3), dtype=np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.line(img, src, dst, color, thickness=1)
    cv2.polylines(img, tx, isClosed=True, color=color, thickness=2)
    plt.figure(figsize=(20, 18))
    plt.imshow(img)
    plt.show()
    return img
def homographyAddTranslation(h, translation):
    h_t = np.eye(3)
    h_t[0, 2] = translation[1]
    h_t[1, 2] = translation[0]
    return h_t @ h

def crop_image(img,percentile=.6):
    height,width=img.shape[:2]
    h=int(percentile*height/2)
    w=int(percentile*width/2)
    return img[h:-h,w:-w]

def grid_img(img,grid_size=60):
    data=img.copy()
    h,w=img.shape[:2]
    n_h=h//grid_size
    n_w=w//grid_size
    skip=-1
    for m in range(n_h-1):
        for n in range(n_w-1):
            skip*=-1
            if skip==1:
                data[m*grid_size:(m+1)*grid_size,n*grid_size:(n+1)*grid_size]=0
    return data
def stichImages(img1, img2, h):
    dst_h, dst_w = img2.shape[:2]
    dst_shape = (dst_w * 3, dst_h * 3)
    h = homographyAddTranslation(h, (dst_h, dst_w))
    img = cv2.warpPerspective(img1, h, dst_shape)
#     fig, ax = plt.subplots(1, 2, figsize=(20, 18))
    padded_img = np.pad(img2, [[dst_h, dst_h], [dst_w, dst_w]])
#     ax[0].imshow(img)
#     ax[1].imshow(padded_img)
#     plt.show()
    img_registeded = padded_img.copy()
    img=grid_img(img)
    img_registeded[img != 0] = img[img != 0]
    img_registeded=crop_image(img_registeded)
    plt.figure(figsize=(14, 10))
    plt.imshow(img_registeded)
    plt.show()
    return


def test_network(img1_rgb, img2_rgb, net, detector):
    img1 = norm_RGB(img1_rgb)[None]
    xys_img1, desc_img1, scores_img1, reliability_img1, repeatability_img1 = extract_multiscale(net, img1, detector,
                                                                                                min_size=64,
                                                                                                verbose=False)
    img2 = norm_RGB(img2_rgb)[None]
    xys_img2, desc_img2, scores_img2, reliability_img2, repeatability_img2 = extract_multiscale(net, img2, detector,
                                                                                                min_size=64,
                                                                                                verbose=False)

    fig, ax = plt.subplots(2, 3, figsize=(20, 8))
    ax[0, 0].imshow(img1_rgb)
    pos = ax[0, 1].imshow(reliability_img1)
    plt.colorbar(pos, ax=ax[0, 1])
    pos = ax[0, 2].imshow(repeatability_img1)
    plt.colorbar(pos, ax=ax[0, 2])
    ax[1, 0].imshow(img2_rgb)
    pos = ax[1, 1].imshow(reliability_img2)
    plt.colorbar(pos, ax=ax[1, 1])
    pos = ax[1, 2].imshow(repeatability_img2)
    plt.colorbar(pos, ax=ax[1, 2])
    plt.show()

    kps1 = xys_img1.cpu().detach().numpy()
    scores1 = scores_img1.cpu().detach().numpy()
    descs1 = desc_img1.cpu().detach().numpy()
    kp1 = np.array([kp for kp, score, desc in zip(kps1, scores1, descs1) if score > reliability_thr])
    desc1 = np.array([desc for kp, score, desc in zip(kps1, scores1, descs1) if score > reliability_thr])

    kps2 = xys_img2.cpu().detach().numpy()
    scores2 = scores_img2.cpu().detach().numpy()
    descs2 = desc_img2.cpu().detach().numpy()
    kp2 = np.array([kp for kp, score, desc in zip(kps2, scores2, descs2) if score > reliability_thr])
    desc2 = np.array([desc for kp, score, desc in zip(kps2, scores2, descs2) if score > reliability_thr])
    fig, ax = plt.subplots(1 ,2, figsize=(20, 10))
    if len(kp1) <2 or len(kp2)< 2: return
    ax[0].imshow(img2_rgb)
    ax[0].scatter(x=kp2[:, 0], y=kp2[:, 1], marker='x', linewidths=1, c='r', s=150)
    ax[1].imshow(img1_rgb)
    ax[1].scatter(x=kp1[:, 0], y=kp1[:, 1], marker='x', linewidths=1, c='r', s=150)
    plt.show()
    keys1 = to_opencv_keypoints(kp1)
    keys2 = to_opencv_keypoints(kp2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1_rgb, keys1, img2_rgb, keys2, good_matches, outImg=None, flags=2)
    plt.figure(figsize=(20, 18))
    plt.imshow(img3)
    pt1 = [kp1[m[0].queryIdx][:2] for m in good_matches]
    pt2 = [kp2[m[0].trainIdx][:2] for m in good_matches]
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    plt.show()
    h, mask = cv2.findHomography(srcPoints=pt1, dstPoints=pt2, method=cv2.FM_RANSAC, ransacReprojThreshold=5)
    imgg = drawMatches(pt1, img1_rgb[...,0], pt2, img2_rgb[..., 0], mask, h)
    plt.show()
    img = cv2.warpPerspective(img1_rgb[...,0], h, img2_rgb[...,0].shape[::-1])
    img_registeded = stichImages(img1_rgb[...,0], img2_rgb[...,0], h)
    plt.show()




# Training loop #
img1 = cv2.imread("/home/javid/SAR_R2D2/testdata/a1.jpg")
img2 = cv2.imread("/home/javid/SAR_R2D2/testdata/a2.jpg")
reliability_thr = .7
repeatability_thr = .7
detector = NonMaxSuppression(rel_thr = reliability_thr, rep_thr = repeatability_thr)

save_path = "./my_models/saved_model.pt"
gpu = -1
# train_data = "ASF"
# train_data = "F"
train_data = "Z"



data_loader = default_dataloader
threads = 1
batch_size = 2
net = default_net
sampler = default_sampler
N = patch_size = 16
loss = default_loss
learning_rate = 1e-3
weight_decay = 5e-3
epochs = 1000
network_path = "./models/faster2d2_WASF_N16.pt"


# iscuda = common.torch_set_gpu(gpu)
iscuda = False
# common.mkdir_for(save_path)


# Create data loader
db = [data_sources[key] for key in train_data]
x = data_loader.replace('`data`',','.join(db)).replace('\n','')
print(x)
db = eval(data_loader.replace('`data`',','.join(db)).replace('\n',''))
print("Training image database =", db)
loader = threaded_loader(db, False, threads, batch_size, shuffle=True)
for a in loader:
    b = a
    break


# net = load_network(network_path)
net = Custom_Quad_L2Net_ConfCFS()
# net = Quad_L2Net_ConfCFS()

# create losses
loss = loss.replace('`sampler`',sampler).replace('`N`',str(patch_size))
print("\n>> Creating loss = " + loss)
loss = eval(loss.replace('\n',''))


# create optimizer
optimizer = optim.Adam( [p for p in net.parameters() if p.requires_grad],
                        lr=learning_rate, weight_decay=weight_decay)

train = MyTrainer(net, loader, loss, optimizer)
if iscuda: train = train.cuda()



for epoch in range(epochs):
    print(f"\n>> Starting epoch {epoch}...")
    train()
    # test_network(img1, img2, net, detector)

print(f"\n>> Saving model to {save_path}")

torch.save({'net': net, 'state_dict': net.state_dict()}, save_path)


x = load_network_custom(save_path)