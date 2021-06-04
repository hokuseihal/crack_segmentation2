import torch.nn as nn
from loss import FocalLoss,DiceLoss

size=(256,256)
criterion=nn.CrossEntropyLoss()
batchsize=8
epochs=500
savefolder='tmp'
cpu=True
CR=True
