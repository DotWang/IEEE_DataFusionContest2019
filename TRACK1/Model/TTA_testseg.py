import argparse
import os
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from mypath import Path
from dataloaders import make_test_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
import tifffile

# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import compute_unary, create_pairwise_bilateral,create_pairwise_gaussian, softmax_to_unary

LAS_LABEL_GROUND = 2
LAS_LABEL_TREES = 5
LAS_LABEL_ROOF = 6
LAS_LABEL_WATER = 9
LAS_LABEL_BRIDGE_ELEVATED_ROAD = 17
LAS_LABEL_VOID = 65

TRAIN_LABEL_GROUND = 0
TRAIN_LABEL_TREES = 1
TRAIN_LABEL_BUILDING = 2
TRAIN_LABEL_WATER = 3
TRAIN_LABEL_BRIDGE_ELEVATED_ROAD = 4
TRAIN_LABEL_VOID = 5

LABEL_MAPPING_LAS2TRAIN = {}
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_GROUND] = TRAIN_LABEL_GROUND
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_TREES] = TRAIN_LABEL_TREES
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_ROOF] = TRAIN_LABEL_BUILDING
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_WATER] = TRAIN_LABEL_WATER
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_BRIDGE_ELEVATED_ROAD] = TRAIN_LABEL_BRIDGE_ELEVATED_ROAD
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_VOID] = TRAIN_LABEL_VOID

LABEL_MAPPING_TRAIN2LAS = {}
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_GROUND] = LAS_LABEL_GROUND
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_TREES] = LAS_LABEL_TREES
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BUILDING] = LAS_LABEL_ROOF
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_WATER] = LAS_LABEL_WATER
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BRIDGE_ELEVATED_ROAD] = LAS_LABEL_BRIDGE_ELEVATED_ROAD

IMG_FILE_STR = '_RGB'
CLSPRED_FILE_STR = '_CLS'
DEPTH_FILE_STR = '_AGL'

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    #type==3 x = (2x-max(x))/(max(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
    elif type==3:
        maxX = np.max(X,0)
        X_norm = 2*X-maxX
        X_norm = X_norm/maxX
    return X_norm    
    
def convert_labels(Lorig, toLasStandard=True):
    """
    Convert the labels from the original CLS file to consecutive integer values starting at 0
    :param Lorig: numpy array containing original labels
    :param params: input parameters from params.py
    :param toLasStandard: determines if labels are converted from the las standard labels to training labels
        or from training labels to the las standard
    :return: Numpy array containing the converted labels
    """
    
    L = Lorig.copy()
    if toLasStandard:
        labelMapping = LABEL_MAPPING_TRAIN2LAS
    else:
        labelMapping = LABEL_MAPPING_LAS2TRAIN
        
    for key,val in labelMapping.items():
        L[Lorig==key] = val
        
    return L

#%%

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='dfc19seg',
                        choices=['pascal', 'coco', 'cityscapes', 'dfc19seg'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=True,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--crf', type=bool, default=False,
                        help='crf post-processing')

    parser.add_argument('--mode', type=str, default='soft',
                        help='voting method')
  
 
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='/data/PreTrainedModel/Seg/8338.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='output_path')
 

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


   
    if args.batch_size is None:
        args.batch_size = 2 

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size


    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    test_loader, nclass = make_test_loader(args, **kwargs)

    # Define network
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    # Using cuda
    model = torch.nn.DataParallel(model)
    patch_replication_callback(model)
    model = model.cuda()
    output_dir = args.output_dir
    # crfoutput_dir = '/data/yonghao.xu/DFC2019/track1/Result/Split-Valcrf/'
    if os.path.exists(output_dir)==False:
        os.mkdir(output_dir)
    # if os.path.exists(crfoutput_dir)==False:
    #     os.mkdir(crfoutput_dir)
    # Resuming checkpoint
   
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        
        if args.cuda:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
       
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    model.eval()

    tbar = tqdm(test_loader, desc='\r')
 
#%%
    for i, sample in enumerate(tbar):
    
        image,name = sample

        image_ori = np.array(image)
        image_ori = image_ori.transpose(0, 2, 3, 1)

        data1 = image_ori.copy()
        data2 = np.rot90(image_ori, 1, (1, 2)).copy()
        data3 = np.rot90(image_ori, 2, (1, 2)).copy()
        data4 = np.rot90(image_ori, 3, (1, 2)).copy()

        image_h = image_ori[:,:,::-1,:].copy()

        data5 = image_h.copy()
        data6 = np.rot90(image_h, 1, (1, 2)).copy()
        data7 = np.rot90(image_h, 2, (1, 2)).copy()
        data8 = np.rot90(image_h, 3, (1, 2)).copy()

        data=np.concatenate([data1,data2,data3,data4,data5,data6,data7,data8],axis=0)
  

        if args.cuda:

            image = image.cuda()

            data = data.transpose(0,3,1,2)#(8,3,H,W)

            data = torch.from_numpy(data).cuda()

        with torch.no_grad():

            output = nn.functional.softmax(model(data), dim=1)#(8,5,H,W)
        
        pred = output.data.cpu().numpy()
    
        if args.mode=='hard':
            label_temp = np.argmax(pred, axis=1)#(8,H,W)

            #############反变换################

            label1 = label_temp[0].copy()#(1,H,W)
            label2 = np.rot90(label_temp[1], -1, (0, 1)).copy()
            label3 = np.rot90(label_temp[2], -2, (0, 1)).copy()
            label4 = np.rot90(label_temp[3], -3, (0, 1)).copy()

            label5 = label_temp[4][:,::-1].copy()
            label6 = np.rot90(label_temp[5], -1, (0, 1))[:,::-1].copy()
            label7 = np.rot90(label_temp[6], -2, (0, 1))[:,::-1].copy()
            label8 = np.rot90(label_temp[7], -3, (0, 1))[:,::-1].copy()

            ##############投票################

            label_mat_3 = np.concatenate([np.expand_dims(label1,0), np.expand_dims(label2,0), np.expand_dims(label3,0), np.expand_dims(label4,0), np.expand_dims(label5,0), np.expand_dims(label6,0), np.expand_dims(label7,0), np.expand_dims(label8,0)], axis=0)#(8,H,W)


            label=np.zeros((1024*1024,))
            label_mat_3 = np.reshape(label_mat_3,8,1024*1024)
            for m in range(1024*1024):
                temp=label_mat_3[:,m]
                label[m]=np.argmax(np.bincount(temp))

            label = np.reshape(label,1024,1024)
            


            '''
            label=np.zeros(label_temp[0].shape)
            for m in range(label_mat_3.shape[1]):
                for n in range(label_mat_3.shape[2]):
                    temp=label_mat_3[:,m,n]
                    label[m,n]=np.argmax(np.bincount(temp))
            '''
            label = np.expand_dims(label,0)   

            
            if args.crf==False:
                for j in range(len(name)):
                    SegoutName = name[j].replace(IMG_FILE_STR, CLSPRED_FILE_STR)
                    Segpred = label[j].astype('uint8')           
                    Segpred = convert_labels(Segpred, toLasStandard=True)
                    tifffile.imsave(os.path.join(output_dir, SegoutName), Segpred, compress=6)
            '''
            
            if args.crf==False:
                for j in range(8):
                    SegoutName = name[0].replace(IMG_FILE_STR, CLSPRED_FILE_STR+repr(j+1))

                    Segpred = label_mat_3[j].astype('uint8')           
                    Segpred = convert_labels(Segpred, toLasStandard=True)
                    tifffile.imsave(os.path.join(output_dir, SegoutName), Segpred, compress=6)
            '''
        elif args.mode=='soft':
            label_temp = pred

            #############反变换################

            label1 = label_temp[0].copy()#(5,H,W)
            label2 = np.rot90(label_temp[1], -1, (1, 2)).copy()
            label3 = np.rot90(label_temp[2], -2, (1, 2)).copy()
            label4 = np.rot90(label_temp[3], -3, (1, 2)).copy()

            label5 = label_temp[4][:,:,::-1].copy()
            label6 = np.rot90(label_temp[5], -1, (1, 2))[:,:,::-1].copy()
            label7 = np.rot90(label_temp[6], -2, (1, 2))[:,:,::-1].copy()
            label8 = np.rot90(label_temp[7], -3, (1, 2))[:,:,::-1].copy()

            ##############投票################

            label_mat_3 = label1+label2+label3+label4+label5+label6+label7+label8#(5,H,W)

            label=np.argmax(label_mat_3,0)

            label = np.expand_dims(label,0)   

            
            if args.crf==False:
                for j in range(len(name)):
                    SegoutName = name[j].replace(IMG_FILE_STR, CLSPRED_FILE_STR)
                    Segpred = label[j].astype('uint8')           
                    Segpred = convert_labels(Segpred, toLasStandard=True)
                    tifffile.imsave(os.path.join(output_dir, SegoutName), Segpred, compress=6)
            
            
        

#%%
if __name__ == "__main__":
   main()
