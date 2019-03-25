import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module,pointnet_sa_module_res

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    ############# PointNet++ 分割部分，先采样，再分组，并分层提取出点云特征######################

    batch_size = point_cloud.get_shape()[0].value
    print(batch_size)
    num_point = point_cloud.get_shape()[1].value
    print(num_point)
    end_points = {}
    l0_xyz = point_cloud[:,:,:3]
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers, 分层特征提取（加SSG策略，single scale grouping）
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    temp1=l1_points
    #print('l1xyz',l1_xyz.shape,'l1point',l1_points.shape,'l1indices',l1_indices.shape)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    temp2=l2_points
    #print('l2xyz', l2_xyz.shape, 'l2point', l2_points.shape, 'l2indices', l2_indices.shape)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    temp3=l3_points
    #print('l3xyz', l3_xyz.shape, 'l3point', l3_points.shape, 'l3indices', l3_indices.shape)
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    #print('l4xyz', l4_xyz.shape, 'l4point', l4_points.shape, 'l4indices', l4_indices.shape)

    # Feature Propagation layers, 对特征进行插值，恢复到之前的点数，最后对每个点进行分类
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l3_points+=temp3
    #print('l3_point',l3_points.shape)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l2_points+=temp2
    #print('l2_point', l2_points.shape)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128,64], is_training, bn_decay, scope='fa_layer3')
    l1_points+=temp1
    #print('l1_point', l1_points.shape)
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')
    #print('l0_point', l0_points.shape)

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    #print('conv1d_1',net.shape)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.9, is_training=is_training, scope='dp1')
    #print('drop',net.shape)
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')##(B,L,num_class)
    #print('conv1d_2',net.shape)

    return net, end_points


def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,8102,3))
        print(inputs.shape)
        _, _ = get_model(inputs, tf.constant(False), 6)
        #print(net)
