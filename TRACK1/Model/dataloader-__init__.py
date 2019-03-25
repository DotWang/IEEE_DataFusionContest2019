
from dataloaders.datasets import dfc19dsm
from torch.utils.data import DataLoader
from mypath import Path
from glob import glob
import os
import numpy as np

def make_data_loader(args, **kwargs):

    if args.dataset == 'dfc19dsm':

        IMG_FILE_STR = '_RGB'
        IMG_FILE_EXT = 'tif'


        wildcard_image = '*%s.%s' % (IMG_FILE_STR, IMG_FILE_EXT)
        glob_path = os.path.join(Path.db_root_dir('dfc19train'), wildcard_image)
        all_files = glob(glob_path)
        file_num=np.arange(len(all_files))
        np.random.shuffle(file_num)

        ix=int(len(all_files)*0.8)

        trn_file=all_files[:ix]
        val_file=all_files[ix:]

        train_set = dfc19dsm.DFCDSM(args, split='train',file=trn_file)
        val_set = dfc19dsm.DFCDSM(args, split='val',file=val_file)

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

def make_test_loader(args, **kwargs):
    
    if args.dataset == 'dfc19dsm':        
        test_set = dfc19dsm.DFCDSM(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)       
        return test_loader, num_class
    else:
        raise NotImplementedError