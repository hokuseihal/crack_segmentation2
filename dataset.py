import torch
import numpy as np
from torchvision.transforms import ToTensor,Compose
import torchvision.transforms as T

import glob
import random
from PIL import Image
import os
import cv2
from util import setcolor
def binary(x,a=.5):
    assert x.shape[0] == 3
    x[x>a]=1
    x[x<a]=0
    return x
class LinerCrackDataset(torch.utils.data.Dataset):
    def __init__(self,txt,size,**kwargs):
        self.size=size
        with open(txt) as f:
            self.txt=[l.strip().replace('jpg','txt') for l in f.readlines() if os.path.exists(l.strip()[:-3]+'txt') and os.path.exists(l.strip()[:-3]+'jpg')]
        print(self.txt)
        self.transform=T.Compose([T.Resize(size),T.ToTensor()])
    def __len__(self):
        return len(self.txt)
    def __getitem__(self,idx):
        im=Image.open(self.txt[idx].replace('txt','jpg'))
        mask=loadtxt(self.txt[idx])
        mask=np.floor(cv2.resize(mask,self.size)).astype(np.int64)
        #save mask
        # maskimg=setcolor(torch.from_numpy(np.expand_dims(mask,axis=0)),torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0]]))[0]//255
        # ToPILImage()(maskimg).save(self.txt[idx].replace('.txt','.jpg'))
        return self.transform(im),torch.from_numpy(mask)
def loadtxt(path):
    thickness=7
    def getdata(ind,sec='point'):
        for d in data:
            # print(d[0],ind)
            if len(d)==5 and d[0]==ind:
                if sec=='point':return int(d[1]),int(d[2])
                elif sec=='cls':return int(d[3])
        # print(f"{path},{ind} is not found.")
    mask=np.zeros((800,800))
    with open(path) as f:
        data=[d.strip().split(',') for d in f.readlines()]
    # print(data)
    for d in data:
        try:
            if d[-1]=='0': continue
            if len(d)==5:
                cv2.line(mask,(int(d[1]),int(d[2])),getdata(d[-1]),color=int(d[3])+1,thickness=thickness)
            elif len(d)==2:
                cv2.line(mask,getdata(d[0]),getdata(d[1]),thickness=thickness,color=getdata(d[0],'cls')+1)
        except:
            # print(f'ERROR on {path},{d}')
            pass
    return mask

if __name__=='__main__':
    linerdataset = LinerCrackDataset('datasets/liner/train.txt', (256,256),path='HI')
    # for i in range(len(linerdataset)):
    #     print(i)