import numpy as np
import pickle
from PIL import Image
import time
import shutil
from random import randint
import argparse
import scipy.io

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ResNet3D_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, nb_per_stack, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.nb_per_stack = nb_per_stack

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.mode == 'train':
            video, nb_clips = self.keys[idx].split('[@]')
            clips_idx = randint(1,int(nb_clips))
        elif self.mode == 'val':
            video,clips_idx = self.keys[idx].split('[@]')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        #print video,clips_idx
        data = get_rgb_clips(video, int(clips_idx), self.nb_per_stack, self.root_dir, self.transform, self.mode)
        
        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (video.split('/',1)[1],data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample

class ResNet3D_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, nb_per_stack, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_path=data_path
        self.nb_per_stack=nb_per_stack
        self.dic_nb_frame={}
        #load data dictionary
        with open(dic_path+'/dic_pose_train.pickle','rb') as f:
            self.dic_training=pickle.load(f)
        f.close()

        with open(dic_path+'/dic_pose_test.pickle','rb') as f:
            self.dic_testing=pickle.load(f)
        f.close()

    def run(self):
        self.get_frame_count()
        self.test_video_segment_labeling()
        self.train_video_labeling()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader
    
    def get_frame_count(self):
        self.frame_count={}
        self.train_video={}
        self.test_video={}
        for key in self.dic_training:
            classname,videoname,frame = key.split('/')
            new_key = classname+'/'+videoname 
            #frame count
            if new_key not in self.frame_count.keys():
                self.frame_count[new_key] = int(frame.split('.',1)[0])
            else:
                if int(frame.split('.',1)[0]) > self.frame_count[new_key]:
                    self.frame_count[new_key] = int(frame.split('.',1)[0])
            #dic[classname/videoname] = label
            if new_key not in self.train_video.keys():
                self.train_video[new_key] = self.dic_training[key]

        for key in self.dic_testing:
            classname,videoname,frame = key.split('/')
            new_key = classname+'/'+videoname
            if new_key not in self.frame_count.keys():
                self.frame_count[new_key] = int(frame.split('.',1)[0])
            else:
                if int(frame.split('.',1)[0]) > self.frame_count[new_key]:
                    self.frame_count[new_key] = int(frame.split('.',1)[0])
            if new_key not in self.test_video.keys():
                self.test_video[new_key] = self.dic_testing[key]
    def test_video_segment_labeling(self):
        self.dic_test_idx = {}
        for video in self.test_video: # dic[video] = label
            nb_frame = int(self.frame_count[video])-self.nb_per_stack
            if nb_frame <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            for clip_idx in range(nb_frame):
                if clip_idx % self.nb_per_stack ==0:
                    key = video + '[@]' + str(clip_idx+1)
                    self.dic_test_idx[key] = self.test_video[video]

    def train_video_labeling(self):
        self.dic_video_train={}
        for video in self.train_video: # dic[video] = label

            nb_clips = self.frame_count[video]-self.nb_per_stack
            if nb_clips <= 0:
                raise ValueError('Invalid nb_per_stack number {} ').format(self.nb_per_stack)
            key = video +'[@]' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]
                            
    def train(self):
        training_set = ResNet3D_dataset(dic=self.dic_video_train, root_dir=self.data_path,
            nb_per_stack=self.nb_per_stack,
            mode='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 
            ]))
        print '==> Training data :',len(training_set),' videos'
        print training_set[1]

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)

        return train_loader

    def val(self):
        validation_set = ResNet3D_dataset(dic= self.dic_test_idx, root_dir=self.data_path ,
            nb_per_stack=self.nb_per_stack,
            mode ='val',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 
            ]))
        print '==> Validation data :',len(validation_set),' clips'
        print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

def get_rgb_clips(key, clip_idx, nb_per_stack, root_dir, transform, mode):
    out=torch.FloatTensor(3,nb_per_stack,112,112)
    classname,videoname = key.split('/')
    index=int(clip_idx)
    #rc = GroupRandomCrop()
    #rc.get_params()
    for i in range(nb_per_stack):
        n = classname+'/'+videoname+'/'+ str(index+i).zfill(5)+'.png'
        img = Image.open(root_dir+n)
        if mode =='train':
            out[:,i,:,:] = transform(img.resize([112,112]))

        elif mode =='val':
            out[:,i,:,:] = transform(img.resize([112,112]))
        else:
            raise ValueError('There are only train and val mode')
            
    return out

'''
class GroupRandomCrop():
    def get_params(self):
        self.h_crop = 112
        self.w_crop = 112
        
        self.h0 = randint(0,128-self.h_crop)
        self.w0 = randint(0,128-self.w_crop)
        

    def crop(self,img):
        crop = img.crop([self.h0,self.w0,self.h0+self.h_crop,self.w0+self.w_crop])
        return crop    
'''
if __name__ == '__main__':
    data_loader = ResNet3D_DataLoader(BATCH_SIZE=1,num_workers=1,
                                        nb_per_stack=15,
                                        dic_path='/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/',
                                        data_path='/home/ubuntu/data/JHMDB/Rename_Images/'
                                        )
    train_loader,val_loader = data_loader.run()
    print type(train_loader),type(val_loader)