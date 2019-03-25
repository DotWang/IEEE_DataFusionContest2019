import os
import sys

import tensorflow as tf
import tf_util_sift as tf_util
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None, feature=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    end_points = {}
    l0_xyz = point_cloud[:,:,:3]
    l0_points = point_cloud
    end_points['l0_xyz'] = l0_xyz

    #################################################### E N C O D E R ######################################################

    ################################################### c0
    c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=0.1, out_channel=64,
                                                                  is_training=is_training, bn_decay=bn_decay,
                                                                  scope='layer0_c0', merge='concat')
    ###c0_10_points(B,N,64)
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(c0_l0_xyz, c0_l0_points, npoint=4096, radius=0.1, nsample=32,
                                                       mlp=[128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer0')
    ###l1_points:(B,4096,128)

    ################################################### c1

    c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=0.2, out_channel=128,
                                                                  is_training=is_training, bn_decay=bn_decay,
                                                                  scope='layer1_c0',merge='concat')
    ###c0_11_points(B,4096,128)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(c0_l1_xyz, c0_l1_points, npoint=1024, radius=0.2, nsample=32,
                                                       mlp=[256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    ###l2_points(B,1024,256)

    #################################################### c2

    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=0.2, out_channel=256,
                                                                  is_training=is_training, bn_decay=bn_decay,
                                                                  scope='layer2_c0',merge='concat')
    ###c0_12_points(B,1024,256)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(c0_l2_xyz, c0_l2_points, npoint=256, radius=0.2, nsample=32,
                                                       mlp=[512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    #l3_points(B,256,512)

    #################################################### c3
    c0_l3_xyz, c0_l3_points, c0_l3_indices = pointSIFT_res_module(l3_xyz, l3_points, radius=0.2, out_channel=512,
                                                                  is_training=is_training, bn_decay=bn_decay,
                                                                  scope='layer3_c0',merge='concat')
    ###c0_13_points(B,256,512)

    l4_xyz, l4_points, l4_indices = pointnet_sa_module(c0_l3_xyz, c0_l3_points, npoint=64, radius=0.2, nsample=32,
                                                       mlp=[1024], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')
    # l4_points(B,64,1024)

    #################################################### c4

    c0_l4_xyz, c0_l4_points, c0_l4_indices = pointSIFT_res_module(l4_xyz, l4_points, radius=0.2, out_channel=1024,
                                                                  is_training=is_training, bn_decay=bn_decay,
                                                                  scope='layer4_c0',merge='concat')
    ###c0_14_points(B,64,1024)

    l5_xyz, l5_points, l5_indices = pointnet_sa_module(c0_l4_xyz, c0_l4_points, npoint=16, radius=0.2, nsample=32,
                                                       mlp=[2048], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')
    # l5_points(B,16,2048)

    #################################################### c5

    # c0_l5_xyz, c0_l5_points, c0_l5_indices = pointSIFT_res_module(l5_xyz, l5_points, radius=0.2, out_channel=2048,
    #                                                               is_training=is_training, bn_decay=bn_decay,
    #                                                               scope='layer5_c0',merge='concat')
    # ###c0_15_points(B,64,2048)
    #
    # l6_xyz, l6_points, l6_indices = pointnet_sa_module(c0_l5_xyz, c0_l5_points, npoint=16, radius=0.2, nsample=32,
    #                                                    mlp=[4096], mlp2=None, group_all=False,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer5')
    # l6_points(B,16,4096)

    #################################################### c6

    # c0_l6_xyz, c0_l6_points, c0_l6_indices = pointSIFT_res_module(l6_xyz, l6_points, radius=0.2, out_channel=4096,
    #                                                               is_training=is_training, bn_decay=bn_decay,
    #                                                               scope='layer6_c0', merge='concat')
    # ###c0_16_points(B,16,4096)
    #
    # gp_xyz, gp_points, gp_indices = pointnet_sa_module(c0_l6_xyz, c0_l6_points, npoint=16, radius=0.2, nsample=32,
    #                                                    mlp=[4096], mlp2=None, group_all=True,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer6')
    # # gp_points(B,1,4096)

    #################################################### D E C O D E R #######################################################

    # ugp_points=tf.expand_dims(gp_points, 1)#(B,1,1,4096)
    #
    # ugp_points=tf_util.conv2d_transpose(ugp_points,4096,[int(l6_points.shape[1]),1],padding='VALID',stride=[1,1],
    #                                     activation_fn=None, bn=True, is_training=is_training, scope='dconv1',
    #                                     bn_decay=bn_decay)
    #
    # ugp_points=tf.squeeze(ugp_points, [2])#(B,16,4096)


    ##################################################### stage 6

    # u6_points_0 = PRRB(l6_xyz, l6_points, is_training, bn_decay,1)
    #
    # # u6_points(B,16,4096)
    #
    # c6_points = PCAB(u6_points_0, ugp_points,is_training, bn_decay,1)
    #
    # # c6_points(B,16,4096)
    #
    # u6_points_1 = PRRB(l6_xyz, c6_points, is_training, bn_decay, 2)

    # u6_points(B,16,4096)

    # l5_points_2 = pointnet_fp_module(l5_xyz, l6_xyz, l5_points, l6_points, [2048], is_training, bn_decay,
    #                                scope='fp_layer6')
    #
    # ##################################################### stage 5
    #
    # u5_points_0 = PRRB(l5_xyz, l5_points, is_training, bn_decay, 3)
    #
    # # u5_points(B,64,2048)
    #
    # c5_points = PCAB(u5_points_0, l5_points_2, is_training, bn_decay, 2)
    #
    # # c5_points(B,64,2048)
    #
    # u5_points_1 = PRRB(l5_xyz, c5_points, is_training, bn_decay, 4)
    #
    # # u5_points(B,64,2048)

    l4_points_2 = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [1024], is_training, bn_decay,
                                   scope='fp_layer5')

    ###################################################### stage 4

    u4_points_0 = PRRB(l4_xyz, l4_points, is_training, bn_decay, 5)

    # u4_points(B,128,1024)

    c4_points = PCAB(u4_points_0, l4_points_2, is_training, bn_decay, 3)

    # c4_points(B,128,1024)

    u4_points_1 = PRRB(l4_xyz, c4_points, is_training, bn_decay, 6)

    # u4_points(B,128,1024)

    l3_points_2 = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, u4_points_1, [512], is_training, bn_decay,
                                     scope='fp_layer4')

    ####################################################### stage 3

    u3_points_0 = PRRB(l3_xyz, l3_points, is_training, bn_decay, 7)

    # u3_points(B,512,512)

    c3_points = PCAB(u3_points_0, l3_points_2, is_training, bn_decay, 4)

    # c3_points(B,512,512)

    u3_points_1 = PRRB(l3_xyz, c3_points, is_training, bn_decay, 8)

    # u3_points(B,512,512)

    l2_points_2 = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, u3_points_1, [256], is_training, bn_decay,
                                     scope='fp_layer3')

    ######################################################## stage 2

    u2_points_0 = PRRB(l2_xyz, l2_points, is_training, bn_decay, 9)

    # u2_points(B,2048,256)

    c2_points = PCAB(u2_points_0, l2_points_2, is_training, bn_decay, 5)

    # c2_points(B,2048,256)

    u2_points_1 = PRRB(l2_xyz, c2_points, is_training, bn_decay, 10)

    # u2_points(B,2048,256)

    l1_points_2 = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, u2_points_1, [128], is_training, bn_decay,
                                     scope='fp_layer2')

    ######################################################### stage 1

    u1_points_0 = PRRB(l1_xyz, l1_points, is_training, bn_decay, 11)

    # u1_points(B,8192,128)

    c1_points = PCAB(u1_points_0, l1_points_2, is_training, bn_decay, 6)

    # c1_points(B,8192,128)

    u1_points_1 = PRRB(l1_xyz, c1_points, is_training, bn_decay, 12)

    # u1_points(B,8192,128)

    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, u1_points_1, [128, 128], is_training, bn_decay,
                                     scope='fp_layer1')

    _, l0_points, _ = pointSIFT_module(l0_xyz, l0_points, radius=0.1, out_channel=128, is_training=is_training,
                                       bn_decay=bn_decay, scope='fsift_0')

    ######################################################### stage 0
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    ###l0_points(B,8192*5,128)
    end_points['feats']=net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    extract_fea=tf.concat([end_points['l0_xyz'],end_points['feats']],axis=-1)

    return net, extract_fea

def get_loss(pred, label, smpw):
    """
    :param pred: BxNxC
    :param label: BxN
    :param smpw: BxN
    :return:
    """
    probas = tf.reshape(pred, (-1, 5))
    labels = tf.reshape(label, (-1,))
    smpws = tf.reshape(smpw, (-1,))
    valid = tf.not_equal(labels, 255)

    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    vsmpws  = tf.boolean_mask(smpws, valid, name='valid_smpws')

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=vlabels, logits=vprobas, weights=vsmpws)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def PRRB(xyz,x_points,is_training, bn_decay, i):

    c=x_points.shape[2]

    # (B,N,512)
    x1 = tf_util.conv1d(x_points, c, 1, padding='VALID', bn=True,
                        is_training=is_training, scope='fc_URB_'+str(i),bn_decay=bn_decay,activation_fn=None)
    x11=x1

    # (B,N,512)
    _, x11, _ = pointSIFT_module(xyz, x11, radius=0.2, out_channel=c, is_training=is_training,
                                       bn_decay=bn_decay,bn=True, scope='fsift_URB_'+str(i)+'_1')

    x11 = lrelu(x11, 0)

    # (B,N,512)
    _, x11, _ = pointSIFT_module(xyz, x11, radius=0.2, out_channel=c, is_training=is_training,
                              bn_decay=None, bn=False, scope='fsiftnobn_URB_'+str(i)+'_1')

    # (B,N,512)
    x=tf.add(x11,x1)

    x = lrelu(x, 0)

    return x

def PCAB(x1,x2,is_training, bn_decay,j):

    # x1, detail feature map, x2, semantic feature map

    x = tf.concat([x1, x2], axis=-1)

    c=x.shape[2]

    # gp_xyz, gp_points, gp_indices = pointnet_sa_module(xyz, x, npoint=16, radius=0.2, nsample=32,
    #                                                    mlp=[c], mlp2=None, group_all=True,
    #                                                    is_training=is_training, bn_decay=bn_decay, scope='layer6')

    x = tf.reduce_mean(x, axis=[1], keep_dims=True, name='gpvgpool_PCAB_'+str(j))  ##(B,1,C)

    x = tf_util.conv1d(x, c, 1, padding='VALID', bn=True, is_training=is_training,
                        scope='fc_PCAB_' + str(j) + '_1',bn_decay=bn_decay,activation_fn=None)

    x = lrelu(x, 0)

    x = tf_util.conv1d(x, x1.shape[2], 1, padding='VALID', bn=True, is_training=is_training,
                       scope='fc_PCAB_' + str(j) + '_2', bn_decay=bn_decay, activation_fn=None)

    x = tf.nn.sigmoid(x)

    x = tf.multiply(x1,x)

    x = tf.add(x,x2)

    return x


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,40960,4))
        net, fea = get_model(inputs, tf.constant(True), 6)
        print(net.shape,fea.shape)






















