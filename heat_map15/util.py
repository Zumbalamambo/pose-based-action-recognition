import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Optimizer

# Dataset
class JHMDB_Pose_heatmap_data_set(Dataset):  
    def __init__(self, dic, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.keys= dic.keys()
        self.values=dic.values()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]

        mat = scipy.io.loadmat(self.root_dir + key)['final_score']
        h,w,c = np.shape(mat)
        heat_map = np.zeros((15,224,224))
        for i in range(c):
            heat_map[i,:,:] = self.transform(Image.fromarray(mat[:,:,i]))


        label = self.values[idx]
        label = int(label)-1
        data=torch.from_numpy(heat_map).float()
        sample = (key, data, label)
        #print sample
        return sample
'''
class JHMDB_Pose_testing_set(Dataset):  
    def __init__(self, dic, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.keys= dic.keys()
        self.values=dic.values()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        key = self.keys[idx]

        mat = scipy.io.loadmat(self.root_dir + key)['final_score']
        h,w,c = np.shape(mat)
        heat_map = np.zeros((224,224,15))
        for i in range(c):
            heat_map[:,:,i] = self.transform(Image.fromarray(mat[:,:,i]))

        label = self.values[idx]
        label = int(label)-1
        heat_map=torch.from_numpy(heat_map)
        sample = (heat_map, label)
        return sample
'''
# other util
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

def save_checkpoint(state, is_best, filename='record/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'record/model_best.pth.tar')
'''
def set_channel(dic, channel):
    dic_stack={}
    for key in dic:
        frame_idx = int(key.split('/')[-1].split('.',1)[0])
        if frame_idx % channel == 0:
            dic_stack[key] = dic[key]

    return dic_stack
'''
def record_info(info,filename,mode):

    if mode =='train':

        result = ('Epoch:{a}',
              'Time {batch_time} '
              'Data {data_time} '
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}'.format(a=info['Epoch'], batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5']
        
    if mode =='test':
        result = ('Epoch{0} ',
              'Time {batch_time} '
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} '.format(info['Epoch'], batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names) 

# Test data loader
if __name__ == '__main__':
    import numpy as np
    data_path='/home/ubuntu/data/JHMDB/pose_estimation/pose_estimation/'
    dic_path='/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/'


    with open(dic_path+'/dic_pose_train.pickle','rb') as f:
        dic_training=pickle.load(f)
    f.close()

    with open(dic_path+'/dic_pose_test.pickle','rb') as f:
        dic_testing=pickle.load(f)
    f.close()

    training_set = JHMDB_Pose_heatmap_data_set(dic=dic_training, root_dir=data_path, transform = transforms.Compose([
            transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            ]))
    
    validation_set = JHMDB_Pose_heatmap_data_set(dic=dic_testing, root_dir=data_path ,transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
            ]))
    
    print training_set[1], validation_set[1]