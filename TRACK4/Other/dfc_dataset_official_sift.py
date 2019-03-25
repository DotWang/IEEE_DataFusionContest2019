import os
import pickle
import sys
import numpy as np


class DFCDataset():
    def __init__(self, root, npoints=8192*5, split='train', log_weighting=False, extra_features=[]):

        self.npoints = npoints
        self.root = root
        self.split = split

        # Dataset size causes memory issues with numpy.save; used pickle instead
        #self.data = np.load(os.path.join(self.root, 'dfc_{}_dataset.npy'.format(split)))
        with open(os.path.join(self.root, 'dfc_{}_dataset.pickle'.format(split)),'rb') as f:########处理后的数据
            self.data = pickle.load(f)
        with open(os.path.join(self.root, 'dfc_{}_metadata.pickle'.format(split)),'rb') as f:#######总体统计信息
            self.metadata = pickle.load(f)
        with open(os.path.join(self.root, 'dfc_{}_labels.pickle'.format(split)),'rb') as f:#######对应label
            self.labels = pickle.load(f)
        
        self.log_weighting = log_weighting
        self.extra_features = extra_features
        self.columns = np.array([0,1,2]+extra_features)
        
        self.M = 5
        
        if split=='train':
            scaling_metadata = self.metadata
        else:
            with open(os.path.join(self.root, 'dfc_train_metadata.pickle'),'rb') as f:
                scaling_metadata = pickle.load(f)#########val/tes也用trn的统计信息
        
        scale = np.sqrt(scaling_metadata['variance'])###标准差
        scale[0:3] = np.sqrt(np.mean(np.square(scale[0:3])))#######XYZ的平均标准差
        self.scale = scale
        self.cls_hist = scaling_metadata['cls_hist']####各类别个数
        
        self.compressed_label_map = scaling_metadata['compressed_label_map']###标签转换，key：2,3.。。 value:0,1,2..
        self.decompress_label_map = scaling_metadata['decompress_label_map']###标签转换，key: 0,1,2.   value：2,3.。。
        
        self.labelweights = np.zeros(self.M, dtype='float32')
        for key, ix in self.compressed_label_map.items():###key:truth label,ix:encode label
            if key == 0:
                continue
            else:
                self.labelweights[ix] = self.cls_hist[key]###5类的weight
        print('样本总数', np.sum(self.labelweights))

        if split=='train':
            self.labelweights = self.labelweights/np.sum(self.labelweights)####各类别样本比例的倒数再开根号作为weight
            self.class_freq = self.labelweights
            if self.log_weighting:
                self.labelweights = 1 / np.log(1.2 + self.labelweights)
            else:
                self.labelweights = (1 / np.log(1.2 + self.labelweights)) * np.array([1, 1.1, 2, 1.5, 2.5])
                #self.labelweights = np.sqrt(1/(self.labelweights+1))
                #self.labelweights = 1 / (self.labelweights + 1e-7)
        else:
            self.labelweights = np.ones(self.M, dtype='float32')
        #self.labelweights[self.compressed_label_map[0]] = 1e-12############unlabeld 类别的权重

    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]

        #print(np.unique(labels))

        valid=labels>0##不为0的样本
        n = point_set.shape[0]

        all_idx=np.arange(n)

        valid_idx=all_idx[valid]

        m=valid_idx.shape[0]

        #print("n,m",n,m)

        if self.npoints < m:
            ixs = np.random.choice(valid_idx,self.npoints,replace=False)
        elif self.npoints == m:
            ixs = np.arange(self.npoints)
        else:
            ixs = np.random.choice(valid_idx,self.npoints,replace=True)

        tmp = point_set[ixs,:]
        point_set = tmp[:,self.columns] / self.scale[self.columns]
        semantic_seg = np.zeros(self.npoints, dtype='int32')
        for i in range(self.npoints):
            semantic_seg[i] = self.compressed_label_map[labels[ixs[i]]]###标签转换，转换为传统标签
        sample_weight = self.labelweights[semantic_seg]
        
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    d = DFCDataset(root = '/data/Public Data/IEEE Data Fusion Contest 2019/DFC2019_track4_trainval/trn_data_folder/baseline_first/',split='val',log_weighting=True,extra_features=[3,4])
    point_set, semantic_seg, sample_weight = d[1]###d:(1718,8192,5)
    print(len(d))
    print(np.unique(point_set[:,4]))
    print(np.unique(semantic_seg))
    print(np.bincount(semantic_seg))
    print(point_set.shape)
    print(semantic_seg.shape)
    print(sample_weight.shape)
    print(d.metadata)
    print('各类别数量：', d.metadata['cls_hist'])
    print('各类别log_weight',d.labelweights)
    print("Scale:"+str(d.scale))
    print("Mapping: "+str(d.decompress_label_map))
    print("Weights: "+str(d.labelweights))
    print("Counts: "+str(d.cls_hist))
    tmp = np.array(list(d.cls_hist.values()),dtype=float)
    print("Frequency: "+str(tmp/np.sum(tmp)))##各类别概率
    print("Length: "+str(len(d.data)))
    exit()


