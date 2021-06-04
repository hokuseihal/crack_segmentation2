from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torchvision.utils import save_image

import core
from dataset import LinerCrackDataset as Dataset
from unet import UNet as Model
from util import setcolor, miouf
from loss import  FocalLoss,DiceLoss


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
            output = model(data)
            loss = criterion(output, target)
            miou = miouf(output, target).item()
            core.addvalue(writer, f'loss:{phase}', loss.item(), e)
            core.addvalue(writer, f'miou:{phase}', miou, e)
            print(f'{e}:{batchidx}/{len(dataloader)}, loss:{loss.item():.4f},miou:{miou:.4f} {phase}')
            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batchidx == 0: save_image(
                    torch.cat([data, setcolor(target, clscolor), setcolor(output.argmax(1), clscolor)], dim=2),
                    f'{savefolder}/{e}.jpg')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--loss', default='CE')

    args = parser.parse_args()
    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    # device='cpu'
    model = Model(out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    clscolor = torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0]])
    criterion = nn.CrossEntropyLoss()
    size = (256,256)
    traindataset = Dataset(txt='datasets/liner/train.txt', size=size)
    valdataset = Dataset(txt='datasets/liner/val.txt', size=size)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=cpu_count())
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=args.batchsize, shuffle=True,
                                                num_workers=cpu_count())
    writer = {}
    startepoch = 0
    savefolder = f'result/{args.savefolder}'
    import os

    os.makedirs(savefolder, exist_ok=True)
    for e in range(startepoch, args.epochs):
        operate('train')
        operate('val')
        core.save(model=model, fol=savefolder, argdic=writer, dic=writer)
