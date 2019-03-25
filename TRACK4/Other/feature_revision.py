import argparse
import copy
from datetime import datetime
from enum import Enum
import glob
import importlib
import json
import logging
import math
import numpy as np
import os
import pickle
from pointset_offical_sift import PointSet
import pprint
from queue import Queue
import subprocess
import sys
import tempfile
import tensorflow as tf
import threading

from sklearn.externals import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#####首先返回当前文件绝对路径，然后去掉文件名，返回目录
#ROOT_DIR = os.path.dirname(BASE_DIR)####返回目录的上一级...\pointnet2
#print(ROOT_DIR)
sys.path.append(BASE_DIR) # model,sys.path: python的搜索模块的路径集
sys.path.append(os.path.join(BASE_DIR, 'models')) # no, really model
sys.path.append(BASE_DIR) # provider
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from tf_utils import provider_sift as provider
# import tf_util
# import pc_util

class InputType(Enum):
    TXT='TXT'
    LAS='LAS'

class OutputType(Enum):
    LABELS='LABELS'
    LAS='LAS'
    BOTH='BOTH'
    
    def __str__(self):
        return self.value

def parse_args(argv):
    #####################外部命令解析#######################
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extra_dims', nargs='+', type=int, default=[3], help='Extra dims')
    parser.add_argument('--input_path', required=True, help='Input point clouds path')
    parser.add_argument('--input_type', type=InputType, choices=list(InputType), default=InputType.TXT)
    parser.add_argument('--feature_path', required=True, help='Input point clouds path')
    parser.add_argument('--output_path', required=True, help='Output path')
    parser.add_argument('--output_type', type=OutputType, choices=list(OutputType), default=OutputType.LABELS)

    ###返回带有关键字及路径的命名空间
    return parser.parse_args(argv[1:])

def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()####初始化日志对象

    logFormatter = logging.Formatter("%(asctime)s %(threadName)s[%(levelname)-3.3s] %(message)s")###当前时间，线程名，文本形式的日志级别，用户输出的消息

    ####output_path+文件名+.log扩展名
    fileHandler = logging.FileHandler(os.path.join(opts.output_path,os.path.splitext(os.path.basename(__file__))[0]+'.log'),mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)####文本形式的日志级别：DEBUG，显示DEBUG及以上级别的，NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("%(threadName)s[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()####将日志打在terminal上
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)####文本形式的日志级别：INFO
    rootLogger.addHandler(consoleHandler)

    rootLogger.level=logging.DEBUG

    logging.debug('Options:\n'+pprint.pformat(opts.__dict__))


# Set global variables
FLAGS = parse_args(sys.argv) ####sys.argv：命令行输入的参数

start_log(FLAGS)

EPOCH_CNT = 0

#####################################################导入SVM,RF等网络模型###################################################

MODEL=joblib.load('/data/DW/DataFusion/dfc2019-master/track4/log_data_folder/model_fusion/test_trn_fea/SVM_model.m')

NUM_CLASSES = 5

LABEL_MAP ={0:2,1:5,2:6,3:9,4:17}

print(LABEL_MAP)

def inference():
    # Generate list of files
    if FLAGS.input_type is InputType.TXT:
        files = glob.glob(os.path.join(FLAGS.input_path,"*_PC3.txt"))#####抓取所有符合要求的扩展名文件
    elif FLAGS.input_type is InputType.LAS:
        files = glob.glob(os.path.join(FLAGS.input_path,"*.las"))
    
    # Setup queues
    input_queue = Queue(maxsize=3)###3线程
    output_queue = Queue(maxsize=3)
    
    # Note: this threading implementation could be setup more efficiently, but it's about 2x faster than a non-threaded version.
    logging.info('Starting threads')
    pre_proc = threading.Thread(target=pre_processor,name='Pre-ProcThread',args=(sorted(files),input_queue))
    pre_proc.start()
    main_proc = threading.Thread(target=main_processor,name='MainProcThread',args=(input_queue,output_queue))
    main_proc.start()
    post_proc = threading.Thread(target=post_processor,name='PostProcThread',args=(output_queue,))
    post_proc.start()

    logging.debug('Waiting for threads to finish')
    pre_proc.join()###join(): 阻塞当前上下文环境的线程
    logging.debug('Joined pre-processing thread')
    main_proc.join()
    logging.debug('Joined main processing thread')
    post_proc.join()
    logging.debug('Joined post-processing thread')
    
    logging.info('Done')


def pre_processor(files, input_queue):
    #######################预处理线程：提取各个batch并进行数据增强##########################

    print(files)

    for file in files:

        logging.info('Loading {}'.format(file))

        pset = PointSet(file)

        _, txt_filename = os.path.split(file)

        points=np.load(os.path.join(FLAGS.feature_path,txt_filename[:-8]+'_FEA_XYZIR.npy'))###直接加载坐标和提取的特征，不需要做预处理，直接分类即可
        data=np.load(os.path.join(FLAGS.feature_path,txt_filename[:-8]+'_FEA.npy'))

        logging.debug('Adding {} to queue'.format(file))
        input_queue.put((pset,data,points))#######写队列
        logging.debug('Added {} to queue'.format(file))
    logging.info('Pre-processing finished')
    input_queue.put(None)
    logging.debug('Pre-processing thread finished')


def main_processor(input_queue, output_queue):

    ########################主处理线程：预测+合并############################

    while True:

        in_data = input_queue.get()  ###读取队列

        if in_data is None:
            break

        logging.info('Processing {}'.format(in_data[0].filename))  ###pset.filename

        feature = in_data[1]###候选点的特征
        fea_xyzir=in_data[2]##候选点的坐标

        pred_lab=MODEL.predict(feature)##从每个txt上提取出来的，传统label：0,1,2,3,4,5

        print(pred_lab.shape)

        # Reshape pred_labels and batch_raw to (BxN,1) and (BxN,5) respectively (i.e. concatenate all point sets in batch together)
        # 合并label

        logging.debug('Adding {} to output queue'.format(in_data[0].filename))
        output_queue.put((in_data[0], fea_xyzir, pred_lab))  ###写队列
        logging.debug('Added {} to output queue'.format(in_data[0].filename))
        input_queue.task_done()

    logging.info('Main processing finished')
    output_queue.put(None)
    logging.debug('Main processing thread finished')



def post_processor(output_queue):

    ##############后处理线程：更新点集+转换标签+保存临时文件+生成新的分类文件


    while True:

        out_data = output_queue.get()  ########读队列

        if out_data is None:
            break
        
        pset = out_data[0]
        all_points = out_data[1]
        all_labels = out_data[2]

        logging.info('Post-processing {}'.format(pset.filename))
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save pset to temp file
            ipath = os.path.join(tmpdir,pset.filename+'_original.las')
            pset.save(ipath)

            # Update pset points
            pset.x = all_points[:,0]
            pset.y = all_points[:,1]
            pset.z = all_points[:,2]
            pset.i = all_points[:,3]
            pset.r = all_points[:,4]
            pset.c = np.array([LABEL_MAP[v] for v in all_labels],dtype='uint8')######标签转换
            #pset.c = all_labels

            print(np.unique(pset.c))

            # Save all classed points to a new file
            cpath = os.path.join(tmpdir,pset.filename+'_candidates.las')
            pset.save(cpath)

            if FLAGS.output_type is OutputType.LABELS:
                opath = os.path.join(tmpdir,pset.filename+'.las')
            else:
                opath = os.path.join(FLAGS.output_path,pset.filename+'.las')

            # Run nearest neighbor voting algorithm to classify original points (pdal pipeline):
            ##注意：投票数是FLAGS.n_angles * 4+1
            ##其中4来自分裂点云（旋转前每个点的重叠子标题的名义数量）
            pipeline = {'pipeline':[
                    ipath,
                    {'type':'filters.neighborclassifier','k':3*4+1,'candidate':cpath}, # Note: number of votes is FLAGS.n_angles*4+1, where 4 comes from splitting the point cloud (nominal number of overlapping subtiles per point before rotations)
                    opath]}
            p = subprocess.run(['/home/sigma_wd/anaconda3/envs/pytorch/bin/pdal','pipeline','-s'],input=json.dumps(pipeline).encode())
            if p.returncode:
                raise ValueError('Failed to run pipeline: \n"'+json.dumps(pipeline)+'"')
            
            if not FLAGS.output_type is OutputType.LAS:
                # Load in updated point cloud, save classification file
                pset2 = PointSet(opath)
                pset2.save_classifications_txt(os.path.join(FLAGS.output_path,pset.filename+'_CLS.txt'))#####生成新的分类文件
            output_queue.task_done()
            logging.debug('Finished {}'.format(pset.filename))
    logging.debug('Post-processing thread finished')


if __name__ == "__main__":
    inference()
