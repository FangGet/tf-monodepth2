from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import layers as tfl

class Net(object):
    def __init__(self,is_training, **config):
        self.is_training = is_training
        self.decay = np.float( config['model']['batch_norm_decay'])
        self.epsilon = np.float(config['model']['batch_norm_epsilon'])
        self.batch_size = np.int(config['model']['batch_size']) 
        self.pose_scale = np.float(config['model']['pose_scale'])
        self.W = np.int(config['dataset']['image_width'])
        self.H= np.int(config['dataset']['image_height'])



    def build_disp_net(self, res18_tc, skips):
        print('Building Depth Decoder Model')
        with tf.variable_scope('depth_decoder', reuse=tf.AUTO_REUSE) as scope:
            end_points_collection = scope.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                activation_fn=tf.nn.elu,
                                outputs_collections=end_points_collection):
                filters = [16, 32, 64, 128, 256]
                # disp 5
                iconv5 = self._conv_reflect(res18_tc, 3, filters[4], 1, 'iconv5')
                iconv5_upsample = tf.image.resize_nearest_neighbor(iconv5, [np.int(self.H / 16), np.int(self.W / 16)])
                iconv5_concat = tf.concat([iconv5_upsample, skips[0]], axis=3)
                upconv5 = self._conv_reflect(iconv5_concat, 3,  filters[4], 1, 'upconv5')

                # disp 4
                iconv4 = self._conv_reflect(upconv5, 3, filters[3], 1, 'iconv4')
                iconv4_upsample = tf.image.resize_nearest_neighbor(iconv4, [np.int(self.H / 8), np.int(self.W / 8)])
                iconv4_concat = tf.concat([iconv4_upsample, skips[1]], axis=3)
                upconv4 = self._conv_reflect(iconv4_concat,3, filters[3], 1, 'upconv4')
                disp4 = self._conv_reflect(upconv4, 3, 1, 1, 'disp4', activation_fn=tf.nn.sigmoid)

                # disp 3
                iconv3 = self._conv_reflect(upconv4,3, filters[2], 1, 'iconv3')
                iconv3_upsample = tf.image.resize_nearest_neighbor(iconv3, [np.int(self.H / 4), np.int(self.W / 4)])
                iconv3_concat = tf.concat([iconv3_upsample, skips[2]], axis=3)
                upconv3 = self._conv_reflect(iconv3_concat,3, filters[2],1, 'upconv3')
                disp3 = self._conv_reflect(upconv3,3, 1, 1, 'disp3', activation_fn=tf.nn.sigmoid)

                # disp 2
                iconv2 = self._conv_reflect(upconv3,3, filters[1],1, 'iconv2')
                iconv2_upsample = tf.image.resize_nearest_neighbor(iconv2, [np.int(self.H / 2), np.int(self.W / 2)])
                iconv2_concat = tf.concat([iconv2_upsample, skips[3]], axis=3)
                upconv2 = self._conv_reflect(iconv2_concat,3, filters[1], 1, 'upconv2')
                disp2 = self._conv_reflect(upconv2,3, 1, 1, 'disp2', activation_fn=tf.nn.sigmoid)

                # disp 1
                iconv1 = self._conv_reflect(upconv2,3, filters[0], 1, 'iconv1')
                iconv1_upsample = tf.image.resize_nearest_neighbor(iconv1, [np.int(self.H), np.int(self.W)])
                iconv1_concat = iconv1_upsample
                upconv1 = self._conv_reflect(iconv1_concat,3, filters[0],1,'upconv1')
                disp1 = self._conv_reflect(upconv1, 3, 1, 1, 'disp1', activation_fn=tf.nn.sigmoid)

                return [disp1, disp2, disp3, disp4]

    def build_pose_net2(self, res18):
        print('Building Pose Decoder Model')
        with tf.variable_scope('pose_decoder', reuse=tf.AUTO_REUSE) as scope:
            end_points_collection = scope.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                res18_concat = self._conv(res18, 1, 256, 1, name='pose_conv0')

                pose_conv1 = self._conv(res18_concat, 3, 256, 1, name='pose_conv1')
                pose_conv2 = self._conv(pose_conv1, 3, 256, 1, name='pose_conv2')
                pose_conv3 = slim.conv2d(pose_conv2, 6, 1, stride=1, scope='pose_conv3', activation_fn=None)

                pose_final = tf.reduce_mean(pose_conv3, [1, 2], keepdims=True)
                pose_final = tf.reshape(pose_final, [self.batch_size, 1, 6])
                pose_final = tf.to_float(self.pose_scale) * pose_final

                return pose_final

    def build_pose_net(self, res18_tp, res18_tc, res18_tn):
        print('Building Pose Decoder Model')
        with tf.variable_scope('pose_decoder', reuse=tf.AUTO_REUSE) as scope:
            end_points_collection = scope.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                res18_tp_ = self._conv(res18_tp, 1, 256, 1, name='res18_tp')
                res18_tc_ = self._conv(res18_tc, 1, 256, 1, name='res18_tc')
                res18_tn_ = self._conv(res18_tn, 1, 256, 1, name='res18_tn')
                res18_concat = tf.concat([res18_tp_, res18_tc_, res18_tn_], axis=3)

                pose_conv1 = self._conv(res18_concat,3, 256, 1, name='pose_conv1')
                pose_conv2 = self._conv(pose_conv1, 3, 256, 1, name='pose_conv2')
                pose_conv3 = slim.conv2d(pose_conv2, 2 * 6, 1, stride=1, scope='pose_conv3', activation_fn=None)

                pose_final = tf.reduce_mean(pose_conv3, [1,2], keepdims=True)
                pose_final = tf.reshape(pose_final, [self.batch_size, 2, 6])
                pose_final = tf.to_float(self.pose_scale) * pose_final

                return pose_final

    def build_resnet18(self, x, prefix=''):
        with tf.variable_scope('{}encoder'.format(prefix), reuse=tf.AUTO_REUSE) as scope:
            end_points_collection = scope.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                biases_initializer=None,
                                activation_fn=None,
                                outputs_collections=end_points_collection):
                print('Building ResNet-18 Model')
                filters = [64, 64, 128, 256, 512]
                kernels = [7, 3, 3, 3, 3]
                strides = [2, 0, 2, 2, 2]

                # conv1
                print('\tBuilding unit: conv1')
                with tf.variable_scope('conv1'):
                    x = self._conv(x, kernels[0], filters[0], strides[0], name='conv1') # [H/2, W/2]
                    x = self._bn(x)
                    x = self._activate(x, type='relu',name='relu1')
                    skip4 = x  # [H/2, W/2]
                    x = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='pool')

                # conv2_x
                x = self._residual_block(x, name='conv2_1')
                x = self._residual_block(x, name='conv2_2')
                skip3 = x # [H/4, W/4]

                # conv3_x
                x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
                x = self._residual_block(x, name='conv3_2')
                skip2 = x # [H/8, W/8]

                # conv4_x
                x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
                x = self._residual_block(x, name='conv4_2')
                skip1 = x # [H/16, W/16]

                # conv5_x
                x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
                x = self._residual_block(x, name='conv5_2')
                # [H/32, W/32]

        return x, [skip1, skip2, skip3, skip4]

    def _residual_block_first(self, x, out_channel, stride, name='unit'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            print('\tBuilding residual unit: {}'.format(scope.name))
            if in_channel == out_channel:
                if stride == 1:
                    short_cut = tf.identity(x)
                else:
                    short_cut = slim.max_pool2d(x, [stride, stride], stride=stride, padding='SAME', scope='pool')
            else:
                short_cut = self._conv(x, 1, out_channel, stride, name='shortcut')
            # residual
            x = self._conv(x, 3, out_channel, stride, name='conv1')
            x = self._bn(x)
            x = self._activate(x, type='relu', name='relu1')
            x = self._conv(x, 3, out_channel, 1, name='conv2')
            x = self._bn(x)

            x = x + short_cut
            x = self._activate(x, type='relu', name='relu2')
            return x

    def _residual_block(self, x, name='unit'):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            print('\tBuildint residual unit: {}'.format(scope.name))
            short_cut = x
            x= self._conv(x, 3, num_channel, 1, name='conv1')
            x = self._bn(x)
            x = self._activate(x,type='relu', name='relu1')
            x = self._conv(x, 3, num_channel, 1, name='conv2')
            x = self._bn(x)

            x = x + short_cut
            x = self._activate(x, type='relu', name='relu2')
            return x

    def _conv(self, x, filter_size, out_channel, stride, pad='SAME', name='conv'):
        x = slim.conv2d(x, out_channel, [filter_size, filter_size], stride,padding=pad, scope=name)
        return x

    def _conv_reflect(self, x, filter_size, out_channel, stride, name='conv', activation_fn=tf.nn.elu):
        pad_size = np.int(filter_size // 2)
        pad_x = tf.pad(x,[[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], mode='REFLECT')
        x = slim.conv2d(pad_x, out_channel, [filter_size, filter_size], stride, padding='VALID', scope=name, activation_fn=activation_fn)
        return x

    def _bn(self, x):
        #x = slim.batch_norm(x,scale=True,decay=self.decay, epsilon=self.epsilon, is_training=True)#, updates_collections=None)
        x = tfl.batch_normalization(x,momentum=self.decay,epsilon=self.epsilon,training=True, name='BatchNorm', fused=True,reuse=tf.AUTO_REUSE) 
        return x

    def _activate(self, x, type='relu', name='relu'):
        if type == 'elu':
            x = tf.nn.elu(x, name=name)
        else:
            x = tf.nn.relu(x, name=name)
        return x
