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
        data =  out = stack_joint_position(video, int(clips_idx), self.nb_per_stack, self.root_dir, self.transform, self.mode)
        
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
            transform=None 
            )
        print '==> Training data :',len(training_set),' videos'
        #print training_set[1]

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
            transform=None 
            )
        print '==> Validation data :',len(validation_set),' clips'
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

def stack_joint_position(key, clip_idx, nb_per_stack, root_dir, transform, mode):
    out=np.zeros((1,nb_per_stack,112,112))
    classname,videoname = key.split('/')
    index=int(clip_idx)
    rc = GroupRandomCrop()
    rc.get_params()
    for i in range(nb_per_stack):
        n = classname+'/'+videoname+'/'+ str(index+i).zfill(5)+'.mat'
        mat = scipy.io.loadmat(root_dir + n)['final_score']
        x0,y0,x1,y1,l=detect_bounding_box(mat)
        if mode =='train':
<<<<<<< HEAD
            data = crop_and_resize(Image.fromarray(mat.sum(axis=2,dtype='uint8')),x0-5,y0-5,x0+l+5,y0+l+5) # 256*256
            out[:,i,:,:] = data

            #out[i,:,:] = random_cropping.crop_and_resize(img=img) # 224*224
        elif mode =='val':
            data = crop_and_resize(Image.fromarray(mat.sum(axis=2,dtype='uint8')),x0-5,y0-5,x0+l+5,y0+l+5) # 256*256
            out[:,i,:,:] = data
=======
            data = crop_and_resize(Image.fromarray(mat.sum(axis=2,dtype='uint8')),x0-5,y0-5,x0+l+5,y0+l+5)
            for j in range(3):
                out[j,i,:,:] = rc.crop(data)
        elif mode =='val':
            data = crop_and_resize(Image.fromarray(mat.sum(axis=2,dtype='uint8')),x0-5,y0-5,x0+l+5,y0+l+5) # 256*256
            for j in range(3):
                out[j,i,:,:] = data.resize([112,112])
>>>>>>> bfe2080adea4d65fb29e492100cce12f64f5ee86
        else:
            raise ValueError('There are only train and val mode')
            

    return torch.from_numpy(out).float()

from numpy import unravel_index
def detect_bounding_box(mat):
    h,w,c = np.shape(mat)
    x0,x1=w,0
    y0,y1=h,0
    for i in range(c):
        a = mat[:,:,i]
        x,y = unravel_index(a.argmax(), a.shape)
        #print x,y
        if x > x1:
            x1=x
        if x < x0:
            x0=x
        if y > y1:
            y1=y
        if y < y0:
            y0=y
    if (x1-x0)>(y1-y0):
        l=x1-x0
    else:
        l=y1-y0

    return x0,y0,x1,y1,l

def crop_and_resize(img,x0,y0,x1,y1):
    crop = img.crop([y0,x0,y1,x1])
    resize = crop.resize([128,128])

    return resize

class GroupRandomCrop():
    def get_params(self):
        '''
        H = [256,224,192,168]
        W = [256,224,192,168]
        id1 = randint(0,len(H)-1)
        id2 = randint(0,len(W)-1)
        '''
        self.h_crop = 112
        self.w_crop = 112
        
        self.h0 = randint(0,128-self.h_crop)
        self.w0 = randint(0,128-self.w_crop)
        

    def crop(self,img):
        crop = img.crop([self.h0,self.w0,self.h0+self.h_crop,self.w0+self.w_crop])
        #resize = crop.resize([224,224])
        return crop    

if __name__ == '__main__':
    data_loader = ResNet3D_DataLoader(BATCH_SIZE=1,num_workers=1,
                                        dic_path='/home/ubuntu/cvlab/pytorch/Sub-JHMDB_pose_stream/get_train_test_split/',
                                        data_path='/home/ubuntu/data/JHMDB/pose_estimation/pose_estimation/'
                                        )
    train_loader,val_loader = data_loader.run()
    print type(train_loader),type(val_loader)