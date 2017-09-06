import numpy as np
import pickle
from PIL import Image
import time
import tqdm
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

from util import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch UCF101 spatial stream video level training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():

    global args
    args = parser.parse_args()

    #Prepare DataLoader
    data_loader = Loader(BATCH_SIZE=arg.batch_size,
                        num_workers=4,
                        data_path=,
                        dic_path=, 
                        )
    
    train_loader = data_loader.train()




class spatial_cnn():

    def __init__(self, model, EPOCH, LR, BATCH_SIZE, resume, evaluate, train_loader, test_loader):

        self.EPOCH=EPOCH
        self.LR=LR
        self.BATCH_SIZE=BATCH_SIZE
        self.resume=resume
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader

    def build_model_and_optimizer(self):
        self.model = network.ResNet18(pretrained= true)
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.LR, momentum=0.9, weight_decay=1e-6)

    def train_1epoch(self):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm.tqdm(self.train_loader)
        for i, (data,label) in enumerate(progress):
    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        record_info = {}
        
        info = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            
        print info

        return

    def validate_1epoch(self):

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        progress = tqdm.tqdm(self.test_loader)
        for i, (data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, label_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



class Loader():
    def __init__(self, BATCH_SIZE, num_workers, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_dir=data_dir

        #load data dictionary
        with open(dic_path+'/sub_jhmdb_train_video.pickle','rb') as f:
            dic_train=pickle.load(f)
        f.close()

        with open(dic_path+'/sub_jhmdb_test_video.pickle','rb') as f:
            dic_test=pickle.load(f)
        f.close()

        self.training_set = JHMDB_rgb_data(dic=dic_training, root_dir=self.data_path, transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        self.validation_set = UCF101_rgb_data(dic=dic_testing, root_dir=self.data_path ,transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
                
    def train(self):
        train_loader = DataLoader(
            dataset=self.training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        test_loader = DataLoader(
            dataset=self.testing_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return test_loader