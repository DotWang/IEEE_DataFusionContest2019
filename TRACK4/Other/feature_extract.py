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
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointSIFT_pointnet', help='Model name [default: pointnet2_sem_seg]')
    parser.add_argument('--extra_dims', nargs='+', type=int, default=[3], help='Extra dims')
    parser.add_argument('--model_path', default='data/results/scannet/log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during inference [default: 16]')
    parser.add_argument('--n_angles', type=int, default=1, help='Number of angles to use to sample image with')
    parser.add_argument('--input_path', required=True, help='Input point clouds path')
    parser.add_argument('--input_type', type=InputType, choices=list(InputType), default=InputType.TXT)
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

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.model) # import network module，导入网络模型



MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
os.system('cp %s %s' % (MODEL_FILE, FLAGS.output_path)) # bkp of model def，拷贝模型到输出文件夹
os.system('cp '+__file__+' %s' % (FLAGS.output_path)) # bkp of train procedure，拷贝该文件到输出文件夹
NUM_CLASSES = 5
COLUMNS = np.array([0,1,2]+FLAGS.extra_dims)
NUM_DIMENSIONS = len(COLUMNS)

with open(os.path.join(os.path.dirname(FLAGS.model_path),'dfc_train_metadata.pickle'),'rb') as f:
    METADATA = pickle.load(f)#######训练数据总体统计信息
SCALE = np.sqrt(METADATA['variance'])
SCALE[0:3] = np.sqrt(np.mean(np.square(SCALE[0:3])))#######之前除过N，为何现在要mean？？？？？？？？
LABEL_MAP = METADATA['decompress_label_map']####key: 0,1,2。。。value：2,3.。。

print(LABEL_MAP)

def inference():
    # Generate list of files
    if FLAGS.input_type is InputType.TXT:
        files = glob.glob(os.path.join(FLAGS.input_path,"*PC3.txt"))#####抓取所有符合要求的扩展名文件
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


def prep_pset(pset):

    ##########################对点集的预处理#########################

    data64 = np.stack([pset.x,pset.y,pset.z,pset.i,pset.r],axis=1)
    offsets = np.mean(data64[:,COLUMNS],axis=0)
    data = (data64[:,COLUMNS]-offsets).astype('float32')###x,y,z减均值？
    
    n = len(pset.x)
    
    if NUM_POINT < n:
        ixs = np.random.choice(n,NUM_POINT,replace=False)########从一个文件挑出NUM_POINT个点？？？？？？？？？？？？
    elif NUM_POINT == n:
        ixs = np.arange(NUM_POINT)
    else:
        ixs = np.random.choice(n,NUM_POINT,replace=True)

    ########原版+标准化之后的
    return data64[ixs,:], data[ixs,:] / SCALE[COLUMNS], pset.c[ixs]


def get_batch(dataset, start_idx, end_idx):

    #######################对某个batch只加权，不做drop###################################

    bsize = end_idx-start_idx
    rsize = min(end_idx,len(dataset))-start_idx
    batch_raw = np.zeros((rsize, NUM_POINT, 5), dtype=np.float64)
    batch_data = np.zeros((bsize, NUM_POINT, NUM_DIMENSIONS), dtype=np.float32)
    batch_lab = np.zeros((bsize,NUM_POINT),dtype=np.uint8)
    for i in range(rsize):
        pset = dataset[start_idx+i]
        batch_raw[i,...], batch_data[i,...], batch_lab[i,...] = prep_pset(pset)###一个小batch

    ########原版+标准化之后的
    return batch_raw, batch_data,batch_lab


def pre_processor(files, input_queue):
    #######################预处理线程：提取各个batch并进行数据增强##########################
    for file in files:
        
        logging.info('Loading {}'.format(file))

        lab_file = file[:-7] + 'CLS.txt'

        pset = PointSet(file,lab_file)
        psets = pset.split()

        num_batches = int(math.ceil((1.0*len(psets))/BATCH_SIZE))###大batch的数量，一个大batch里边有若干小batch

        data = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            for k in range(FLAGS.n_angles):
                batch_raw, batch_data, batch_lab = get_batch(psets, start_idx, end_idx)###一个大batch
    
                if k == 0:
                    aug_data = batch_data
                else:
                    ang = (1.0*k)/(1.0*FLAGS.n_angles) * 2 * np.pi
                    if FLAGS.extra_dims:
                        aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:,:,0:3]),
                                batch_data[:,:,3:]),axis=2)#########数据增强：随机旋转点云+连接其他维度
                    else:
                        aug_data = provider.rotate_point_cloud_z(batch_data)
                
                data.append((batch_raw,aug_data,batch_lab))
        
        logging.debug('Adding {} to queue'.format(file))
        input_queue.put((pset,data))#######写队列
        logging.debug('Added {} to queue'.format(file))
    logging.info('Pre-processing finished')
    input_queue.put(None)
    logging.debug('Pre-processing thread finished')


def main_processor(input_queue, output_queue):

    ########################主处理线程：预测+合并############################

    with tf.Graph().as_default():
        with tf.device('/device:GPU:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)####点云，标签，权重
            is_training_pl = tf.placeholder(tf.bool, shape=())##控制BN状态
            
            logging.info("Loading model")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)####网络的输出

            print('==========================================',end_points.shape)

            # feature = graph.get_operation_by_name('fc1').outputs[0]
            # print('=========================================',feature.shape)

            logging.info(pred.shape)
            saver = tf.train.Saver()
        
        # Create a session，session相关设置
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.，加载模型去预测
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'labels_pl':labels_pl,
               'endpoint_fea':end_points}
        is_training = False
        logging.info("Model loaded")
    
        while True:
            in_data = input_queue.get()###读取队列
            if in_data is None:
                break
            
            logging.info('Processing {}'.format(in_data[0].filename))###pset.filename
            batch_list = in_data[1]
            for k in range(len(batch_list)):
                batch_raw = batch_list[k][0]
                aug_data  = batch_list[k][1]
                batch_lab = batch_list[k][2]
                
                feed_dict = {ops['pointclouds_pl']: aug_data,
                            ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
                end_points = sess.run([ops['endpoint_fea']], feed_dict=feed_dict)
        
                pred_labels = np.argmax(pred_val[0], 2)# BxN，test的预测类别

                pred_feature=np.array(end_points)

                trn_label=np.array(labels_pl)

                # subset to true batch size as necessary
                if batch_raw.shape[0] != BATCH_SIZE:
                    pred_labels = pred_labels[0:batch_raw.shape[0], :]

                    pred_feature=pred_feature[:,0:batch_raw.shape[0],:,:]#(1,B,N,D)
                    batch_raw = batch_raw[0:batch_raw.shape[0]]
                    batch_lab = batch_lab[0:batch_raw.shape[0],:]

                    print(pred_feature.shape, batch_lab.shape,batch_raw.shape)


                # Reshape pred_labels and batch_raw to (BxN,1) and (BxN,5) respectively (i.e. concatenate all point sets in batch together)
                # 合并label
                #print('+++++++++++++++++',pred_feature.shape[0],pred_feature.shape[1])
                pred_feature=np.reshape(pred_feature,[-1,128])
                batch_lab=np.reshape(batch_lab,[-1,1])
                batch_raw= np.reshape(batch_raw,[-1,5])
                
                if k==0:
                    all_pred = pred_feature
                    all_points = batch_raw
                    all_label = batch_lab
                else:
                    # Concatenate all pointsets across all batches together
                    all_pred = np.concatenate((all_pred,pred_feature),axis=0)
                    all_points = np.concatenate((all_points,batch_raw),axis=0)
                    all_label = np.concatenate((all_label,batch_lab),axis=0)

            logging.debug('Adding {} to output queue'.format(in_data[0].filename))
            output_queue.put((in_data[0],all_points,all_pred,all_label))###写队列
            logging.debug('Added {} to output queue'.format(in_data[0].filename))
            input_queue.task_done()
        logging.info('Main processing finished')
        output_queue.put(None)
    logging.debug('Main processing thread finished')


def post_processor(output_queue):

    ##############后处理线程：更新点集+转换标签+保存临时文件+生成新的分类文件

    while True:
        out_data = output_queue.get()########读队列
        if out_data is None:
            break
        
        pset = out_data[0]
        all_points = out_data[1]
        all_pred = out_data[2]
        all_label= out_data[3]

        print(all_pred.shape)

        np.save(os.path.join(FLAGS.output_path, pset.filename + '_FEA_XYZIR.npy'), all_points)
        np.save(os.path.join(FLAGS.output_path, pset.filename + '_FEA.npy'),all_pred)
        np.save(os.path.join(FLAGS.output_path, pset.filename + '_FEA_LAB.npy'), all_label)

    logging.debug('Post-processing thread finished')


if __name__ == "__main__":
    inference()
