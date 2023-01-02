import os, pdb
import torch
import torch.optim as optim

from tools import common, trainer
from tools.dataloader import *
from nets.patchnet import *
from nets.losses import *
from PIL import Image, ImageOps
from tqdm.notebook import tqdm


from  datasets import *

import matplotlib.pyplot as plt



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

# default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
#                             subd_neg=-8,maxpool_pos=True)"""

default_sampler = """DoubleDescNghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True, desc_dim=128)"""



default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""


data_sources = dict(
    D = toy_db_debug,
    W = db_web_images,
    A = db_aachen_images,
    F = db_aachen_flow,
    S = db_aachen_style_transfer,
    X = db_sar_images,
    Z = db_sar_flow
    )


class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """
    def forward_backward(self, inputs):
        output = self.net(imgs=[inputs.pop('img1'),inputs.pop('img2')])
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details




def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


save_path = "./trained_models"
gpu = -1
# train_data = "ASF"
train_data = "Z"
# train_data = "F"


data_loader = default_dataloader
threads = 1
batch_size = 4
net = default_net
sampler = default_sampler
N = patch_size = 16
loss = default_loss
learning_rate = 1e-4
weight_decay = 5e-4
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

# create losses
loss = loss.replace('`sampler`',sampler).replace('`N`',str(patch_size))
print("\n>> Creating loss = " + loss)
loss = eval(loss.replace('\n',''))


# create optimizer
optimizer = optim.Adam( [p for p in net.parameters() if p.requires_grad],
                        lr=learning_rate, weight_decay=weight_decay)

train = MyTrainer(net, loader, loss, optimizer)
if iscuda: train = train.cuda()



# Training loop #
for epoch in range(epochs):
    print(f"\n>> Starting epoch {epoch}...")
    train()

print(f"\n>> Saving model to {save_path}")

torch.save({'net': net, 'state_dict': net.state_dict()}, save_path)


