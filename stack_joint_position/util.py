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
    def __init__(self, dic, nb_per_stack, root_dir, transform=None):
        #print('==> get total frame numbers of each video')
        frame_count={}
        for key in dic:
            classname,videoname,frame = key.split('/')
            if videoname not in frame_count.keys():
                frame_count[videoname] = int(frame.split('.',1)[0])
            else:
                if int(frame.split('.',1)[0]) > frame_count[videoname]:
                    frame_count[videoname] = int(frame.split('.',1)[0])
        #print frame_count
        print('==> generate new dic for {} stack joint position').format(nb_per_stack)
        dic_stack={}
        for key in dic:
            classname,videoname,frame = key.split('/')
            if int(frame.split('.',1)[0]) < frame_count[videoname]-nb_per_stack:
                dic_stack[key] = dic[key]

        self.root_dir = root_dir
        self.transform = transform
        self.nb_per_stack=nb_per_stack
        self.keys = dic_stack.keys()
        self.values =dic_stack.values() 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]
        out = stack_joint_position(key,self.nb_per_stack, self.root_dir, self.transform)

        label = self.values[idx]
        label = int(label)-1
        data=torch.from_numpy(out).float()
        sample = (key, data, label)
        return sample

def stack_joint_position(key, nb_per_stack, root_dir, transform):
    out=np.zeros((nb_per_stack,224,224))
    classname,videoname,frame = key.split('/')
    index=int(frame.split('.',1)[0])
    for i in range(nb_per_stack):
        n = classname+'/'+videoname+'/'+ str(index+i).zfill(5)+'.mat'
        mat = scipy.io.loadmat(root_dir + n)['final_score']
        out[i,:,:] = transform(Image.fromarray(mat.sum(axis=2,dtype='uint8')))
    
    return out



        




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

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))      
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5']
        
    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
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

    training_set = JHMDB_Pose_heatmap_data_set(dic=dic_training, root_dir=data_path, nb_per_stack=10, transform = transforms.Compose([
            transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            ]))
    
    validation_set = JHMDB_Pose_heatmap_data_set(dic=dic_testing, root_dir=data_path, nb_per_stack=10,transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
            ]))
    print type(training_set[1][1][1,:,:].numpy())
    a = (training_set[1][1][1,:,:].numpy())
    with open('test.pickle','wb') as f:
        pickle.dump(a,f)
    f.close()
