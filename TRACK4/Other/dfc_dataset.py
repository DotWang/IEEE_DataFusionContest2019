import os
import pickle
import sys
import numpy as np


class DFCDataset():
    def __init__(self, root, npoints=8192, split='train', log_weighting=False, extra_features=[]):

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
        
        self.M = 6
        
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
        for key, ix in self.compressed_label_map.items():
            if key == 0:
                self.labelweights[ix] = 0
            else:
                self.labelweights[ix] = self.cls_hist[key]
        print('样本总数', np.sum(self.labelweights))

        if split=='train':
            self.labelweights = self.labelweights/np.sum(self.labelweights)####各类别样本比例的倒数再开根号作为weight
            self.class_freq = self.labelweights
            if self.log_weighting:
                self.labelweights = 1 / np.log(1.2 + self.labelweights)
                #self.labelweights = np.array([1,1,1,1,1,1])
            else:
                self.labelweights = (1 / np.log(1.2 + self.labelweights)) * np.array([1, 1, 1, 1, 3, 3])
                #self.labelweights = np.sqrt(1/(self.labelweights+1))
                #self.labelweights = 1 / (self.labelweights + 1e-7)
        else:
            self.labelweights = np.ones(self.M, dtype='float32')
        self.labelweights[self.compressed_label_map[0]] = 1e-12############unlabeld 类别的权重

    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]
        n = point_set.shape[0]

        #各类别真实行号，从小到大
        unlabeld_idx = np.where(labels == 0)[0]
        ground_idx   = np.where(labels == 2)[0]
        tree_idx = np.where(labels == 5)[0]
        building_idx = np.where(labels == 6)[0]
        water_idx = np.where(labels == 9)[0]
        road_idx = np.where(labels == 17)[0]

        #打乱行号
        np.random.shuffle(unlabeld_idx)
        np.random.shuffle(ground_idx )
        np.random.shuffle(tree_idx)
        np.random.shuffle(building_idx)
        np.random.shuffle(water_idx)
        np.random.shuffle(road_idx)

        #各类别个数
        n_unlabeld=unlabeld_idx.shape[0]
        n_ground=ground_idx.shape[0]
        n_tree=tree_idx.shape[0]
        n_building=building_idx.shape[0]
        n_water=water_idx.shape[0]
        n_road=road_idx.shape[0]
        #print('water,road',n_water,n_road)

        label_list=np.unique(labels)####该batch内的点的种类
        Flag=0

        if label_list.shape[0]>1:

            num_water=num_road=num_tree=num_building=num_ground=num_unlabeld=0
            tr_idx_water=tr_idx_road=tr_idx_tree=tr_idx_building=tr_idx_ground=tr_idx_unlabeld=np.array([])

            for i in label_list[::-1]:

                if i==9:
                    num_water=round(self.npoints*0.1*n_water/(n_water+n_road + 1e-7))
                    se_idx_water=np.random.choice(n_water,num_water,replace=True)
                    tr_idx_water=water_idx[se_idx_water]##前3584个的一部分真实行号（行号已打乱）

                if i==17:
                    num_road = round(self.npoints*0.1* n_road / (n_water + n_road+1e-7))
                    se_idx_road = np.random.choice(n_road, num_road, replace=True)
                    tr_idx_road = road_idx[se_idx_road]

                if i==5:
                    num_tree=round(self.npoints*0.4*n_tree/(n_tree+n_building+1e-7))
                    se_idx_tree=np.random.choice(n_tree, num_tree, replace=True)
                    tr_idx_tree=tree_idx[se_idx_tree]

                if i==6:
                    num_building=round(self.npoints*0.4*n_building/(n_tree+n_building+1e-7))
                    se_idx_building=np.random.choice(n_building,num_building,replace=True)
                    tr_idx_building=building_idx[se_idx_building]

                if i==2:
                    num_ground=round(self.npoints*0.5*n_ground/(n_ground+n_unlabeld+1e-7))
                    se_idx_ground=np.random.choice(n_ground,num_ground,replace=True)
                    tr_idx_ground=ground_idx[se_idx_ground]

            if n_unlabeld>0:
                num_unlabeld=self.npoints-num_water-num_road-num_tree-num_building-num_ground
                se_idx_unlabeld=np.random.choice(n_unlabeld,num_unlabeld,replace=True)
                tr_idx_unlabeld = unlabeld_idx[se_idx_unlabeld]

            ixs=np.concatenate([tr_idx_unlabeld,tr_idx_ground,tr_idx_tree,tr_idx_building,tr_idx_water,tr_idx_road],axis=0).astype(int)
            if ixs.shape[0]<self.npoints:
                Flag=1

        if label_list.shape[0]==1 or Flag==1:
            ixs = np.random.choice(n, self.npoints, replace=False)
            #print(ixs.shape[0])

        #print(ixs.shape)

        # if self.npoints < n:
        #     ixs = np.random.choice(n,self.npoints,replace=False)
        # elif self.npoints == n:
        #     ixs = np.arange(self.npoints)
        # else:
        #     ixs = np.random.choice(n,self.npoints,replace=True)

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
    d = DFCDataset(root = '/data/Public Data/IEEE Data Fusion Contest 2019/DFC2019_track4_trainval/trn_data_folder/baseline_first/',split='train',log_weighting=True,extra_features=[3,4])
    point_set, semantic_seg, sample_weight = d[100]###d:(1718,8192,5)
    print(len(d))
    print(np.unique(point_set[:,4]))
    print(np.unique(semantic_seg))
    print(np.bincount(semantic_seg))
    print(point_set.shape)
    print(semantic_seg.shape)
    print(sample_weight.shape)
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


