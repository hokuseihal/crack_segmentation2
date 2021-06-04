import os
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import core
from dataset import LinerCrackDataset as Dataset
from setting import *
from unet import UNet as Model
from util import setcolor, miouf, cutmix


def operate(phase):
    if phase == 'train':
        dataloader = traindataloader
        model.train()
    else:
        dataloader = valdataloader
        model.eval()
    with torch.set_grad_enabled(phase == 'train'):
        for batchidx, (data, target) in enumerate(dataloader):
            B, C, H, W = data.shape

            data = data.to(device)
            target = target.to(device)
            if CR and batchidx%2==0:
                x_cutmix, cutparam = cutmix(data)
                y_pred_cutmix, _ = cutmix(model(data), cutparam)
                CRloss = F.mse_loss(model(x_cutmix), y_pred_cutmix)
                # segloss = criterion(y_pred_cutmix, cutmix(target, cutparam))
                loss = CRloss
                print(f'CRloss:{CRloss}')
                core.addvalue(writer, f'CRloss:{phase}', CRloss.item(), e)
            else:
                # NORMAL
                output = model(data)
                loss = criterion(output, target)
                miou = miouf(output, target).item()
                core.addvalue(writer, f'miou:{phase}', miou, e)
                print(f'miou:{miou:.4f}')
            core.addvalue(writer, f'loss:{phase}', loss.item(), e)
            print(f'{e}:{batchidx}/{len(dataloader)}, loss:{loss.item():.4f} {phase}')
            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batchidx == 0: save_image(
                    torch.cat([data, setcolor(target, clscolor), setcolor(output.argmax(1), clscolor)], dim=2),
                    f'{savefolder}/{e}.jpg')


if __name__ == '__main__':

    device = 'cuda' if (torch.cuda.is_available() and not cpu) else 'cpu'
    # device='cpu'
    model = Model(out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    clscolor = torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0]])
    criterion = nn.CrossEntropyLoss()
    traindataset = Dataset(txt='datasets/liner/train.txt', size=size)
    valdataset = Dataset(txt='datasets/liner/val.txt', size=size)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                  num_workers=cpu_count())
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=batchsize, shuffle=True,
                                                num_workers=cpu_count())
    writer = {}
    startepoch = 0
    savefolder = f'result/{savefolder}/'

    os.makedirs(savefolder, exist_ok=True)
    savefolder += f'{len(os.listdir(savefolder))}'
    print(savefolder)

    os.makedirs(savefolder, exist_ok=True)
    for e in range(startepoch, epochs):
        operate('train')
        operate('val')
        core.save(model=model, fol=savefolder, argdic=writer, dic=writer)
