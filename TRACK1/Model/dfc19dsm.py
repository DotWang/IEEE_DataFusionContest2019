import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from glob import glob
import tifffile
import torch

class DFCDSM(data.Dataset):

    NUM_CLASSES = 5
    #MINMAX = np.array(((0.,5.),(0.,25.),(0.,50.),(0.,8.),(0.,30.)))
    IGNORE_VALUE = -100. 
    #BIAS = 0.
    CLASS_FILE_STR = '_CLS'
    DEPTH_FILE_STR = '_AGL'
    IMG_FILE_STR = '_RGB'
    IMG_FILE_EXT = 'tif'
    LABEL_FILE_EXT = IMG_FILE_EXT
    def __init__(self, args, split="train",file=None):

        self.split = split
        self.args = args
        self.files = {}

        if split == "train":            
            self.files[split]=file
        elif split == "val":            
            self.files[split] = file
        elif split == "test":
            self.root = Path.db_root_dir('dfc19test')
            wildcard_image = '*%s.%s' % (self.IMG_FILE_STR, self.IMG_FILE_EXT)
            glob_path = os.path.join(self.root, wildcard_image)
            self.files[split] = glob(glob_path)

        self.void_classes = [65]
        self.valid_classes = [2, 5, 6, 9, 17]
        self.class_names = ['Ground', 'Vegetation', 'Building', 'Water', 'Elevated_Road']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.root))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        if self.split == 'train':            
            img_path = self.files[self.split][index]
            seg_path = img_path.replace(self.IMG_FILE_STR,self.CLASS_FILE_STR)
            lbl_path = img_path.replace(self.IMG_FILE_STR,self.DEPTH_FILE_STR)

            _img = Image.fromarray(tifffile.imread(img_path))
            _seg = tifffile.imread(seg_path)
            _seg = self.encode_segmap(_seg)
            _target = tifffile.imread(lbl_path)
            _target[np.isnan(_target)] = self.IGNORE_VALUE

            '''
            for i in range(self.NUM_CLASSES):
                _target[(_seg==i)&(_target<self.MINMAX[i,0])] = self.IGNORE_VALUE
                _target[(_seg==i)&(_target>self.MINMAX[i,1])] = self.IGNORE_VALUE
            _target += self.BIAS
            _target[_target<0] = self.IGNORE_VALUE
            '''

            _target = Image.fromarray(_target)

            sample = {'image': _img, 'label': _target}
            return self.transform_tr(sample)
        
        elif self.split == 'val':
            img_path = self.files[self.split][index]
            seg_path = img_path.replace(self.IMG_FILE_STR,self.CLASS_FILE_STR)
            lbl_path = img_path.replace(self.IMG_FILE_STR,self.DEPTH_FILE_STR)

            _img = Image.fromarray(tifffile.imread(img_path))
            _seg = tifffile.imread(seg_path)
            _seg = self.encode_segmap(_seg)
            _target = tifffile.imread(lbl_path)
            _target[np.isnan(_target)] = self.IGNORE_VALUE

            '''
            for i in range(self.NUM_CLASSES):
                _target[(_seg==i)&(_target<self.MINMAX[i,0])] = self.IGNORE_VALUE
                _target[(_seg==i)&(_target>self.MINMAX[i,1])] = self.IGNORE_VALUE
            _target += self.BIAS
            _target[_target<0] = self.IGNORE_VALUE
            '''


            _target = Image.fromarray(_target)

            sample = {'image': _img, 'label': _target}
            return self.transform_val(sample)

        elif self.split == 'test':
            img_path = self.files[self.split][index]
            _name = self.files[self.split][index].split('/')[-1]
            _img = Image.fromarray(tifffile.imread(img_path))
            
            return self.transform_ts(_img),_name
        #img_path = self.files[self.split][index]
        #lbl_path = img_path.replace(self.IMG_FILE_STR,self.CLASS_FILE_STR)

        #_img = Image.fromarray(tifffile.imread(img_path))
        #_target = tifffile.imread(lbl_path)
        #_target = Image.fromarray(self.encode_segmap(_target))

        #sample = {'image': _img, 'label': _target}
      

        #if self.split == 'train':
        #    return self.transform_tr(sample)
        #elif self.split == 'val':
        #    return self.transform_val(sample)
        #elif self.split == 'test':
        #    return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask



    def transform_tr(self, sample):
        
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            #tr.FixedResize(size=self.args.crop_size),
            #tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        img=composed_transforms(sample)

        data=img['image']
        label=img['label']

        p = np.random.rand(1)[0]

        if p < 0.25:
            data = np.rot90(data, 1, (0, 1)).copy()
            label=np.rot90(label, 1, (0, 1)).copy()
        elif p >= 0.25 and p < 0.5:
            data = np.rot90(data, 2, (0, 1)).copy()
            label = np.rot90(label, 2, (0, 1)).copy()
        elif p >= 0.5 and p < 0.75:
            data = np.rot90(data, 3, (0, 1)).copy()
            label = np.rot90(label, 3, (0, 1)).copy()

        data = torch.from_numpy(data.transpose(2, 0, 1))
        label = torch.from_numpy(label)

        return {'image':data,'label':label}

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):
        
        composed_transforms = transforms.Compose([
            #transforms.Resize(size=self.args.crop_size),            
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 1024
    args.crop_size = 1024

    cityscapes_train = DFCDSM(args, split='train')

    if split=='train':
        dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=1,drop_last=True)
    else:
        dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=1)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

