import pickle,os
from PIL import Image
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='../save/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../save/model_best.pth.tar')

class JHMDB_rgb_data(Dataset):  
    def __init__(self, dic, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.keys= dic.keys()
        self.values=dic.values()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        frame = self.keys[idx] 
        img = Image.open(self.root_dir + frame)
        label = self.values[idx]
        label = int(label)-1

        transformed_img = self.transform(img)
        sample = (transformed_img,label)
                 
        img.close()
        return sample