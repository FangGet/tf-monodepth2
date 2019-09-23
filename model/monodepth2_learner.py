from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from dataloader.data_loader import DataLoader
from model.net import Net
from utils.tools import *

import matplotlib as mpl
import matplotlib.cm as cm

class MonoDepth2Learner(object):
    def __init__(self, **config):
        self.config = config
        self.preprocess = self.config['dataset']['preprocess']
        self.min_depth = np.float(self.config['dataset']['min_depth'])
        self.max_depth = np.float(self.config['dataset']['max_depth'])
        self.root_dir = self.config['model']['root_dir']
        self.pose_type = self.config['model']['pose_type']

    def preprocess_image(self, image):
        image = (image - 0.45) / 0.225
        return image

    def compute_reprojection_loss(self, reproj_image, tgt_image):
        l1_loss = tf.reduce_mean(tf.abs(reproj_image-tgt_image), axis=3, keepdims=True)

        ssim_loss = tf.reduce_mean(self.SSIM(reproj_image, tgt_image), axis=3, keepdims=True)

        loss = self.ssim_ratio * ssim_loss + (1 - self.ssim_ratio) * l1_loss
        #loss = l1_loss

        return loss

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_smooth_loss(self, disp, img):
        disp = disp / ( tf.reduce_mean(disp, [1, 2], keepdims=True) + 1e-7)

        grad_disp_x = tf.abs(disp[:, :-1, :, :] - disp[:, 1:, :, :])
        grad_disp_y = tf.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
        grad_img_y = tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        weight_x = tf.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
        weight_y = tf.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    def build_test(self, build_type='both'):
        self.loader = DataLoader(trainable=False, **self.config)
        self.num_scales = self.loader.num_scales
        self.num_source = self.loader.num_source
        with tf.name_scope('data_loading'):
            self.tgt_image = tf.placeholder(tf.uint8, [self.loader.batch_size,
                    self.loader.img_height, self.loader.img_width, 3])
            tgt_image = tf.image.convert_image_dtype(self.tgt_image, dtype=tf.float32)
            tgt_image_net = self.preprocess_image(tgt_image)
            if build_type != 'depth':
                self.src_image_stack = tf.placeholder(tf.uint8, [self.loader.batch_size,
                    self.loader.img_height, self.loader.img_width, 3 * self.num_source])
                src_image_stack = tf.image.convert_image_dtype(self.src_image_stack, dtype=tf.float32)
                src_image_stack_net = self.preprocess_image(src_image_stack)
            #if self.preprocess:


            #else:
            #    tgt_image_net = tgt_image
            #    src_image_stack_net = src_image_stack

        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)
            pred_disp_rawscale = [tf.image.resize_bilinear(pred_disp[i], [self.loader.img_height, self.loader.img_width]) for i in
                range(self.num_scales)]
            pred_depth_rawscale = disp_to_depth(pred_disp_rawscale, self.min_depth, self.max_depth)

            self.pred_depth = pred_depth_rawscale[0]
            self.pred_disp = pred_disp_rawscale[0]

            if build_type != 'depth':
                num_source = np.int(src_image_stack_net.get_shape().as_list()[-1] // 3)
                assert num_source == 2

                if self.pose_type == 'seperate':
                    res18_ctp, _ = net_builder.build_resnet18(
                        tf.concat([tgt_image_net,src_image_stack_net[:, :, :, :3]], axis=3),
                        prefix='pose_'
                    )
                    res18_ctn, _ = net_builder.build_resnet18(
                        tf.concat([tgt_image_net, src_image_stack_net[:, :, :, 3:]], axis=3),
                        prefix='pose_'
                    )
                elif self.pose_type == 'shared':
                    res18_tp, _ = net_builder.build_resnet18(src_image_stack_net[:, :, :, :3])
                    res18_tn, _ = net_builder.build_resnet18(src_image_stack_net[:, :, :, 3:])
                    res18_ctp = tf.concat([res18_tc, res18_tp], axis=3)
                    res18_ctn = tf.concat([res18_tc, res18_tn], axis=3)
                else:
                    raise NotImplementedError

                pred_pose_ctp = net_builder.build_pose_net2(res18_ctp)
                pred_pose_ctn = net_builder.build_pose_net2(res18_ctn)

                pred_poses = tf.concat([pred_pose_ctp, pred_pose_ctn], axis=1)

                self.pred_poses = pred_poses


    def build_train(self):
        self.ssim_ratio = np.float(self.config['model']['reproj_alpha'])
        self.smoothness_ratio = np.float(self.config['model']['smooth_alpha'])
        self.start_learning_rate = np.float(self.config['model']['learning_rate'])
        self.total_epoch = np.int(self.config['model']['epoch'])
        self.beta1 = np.float(self.config['model']['beta1'])
        self.continue_ckpt = self.config['model']['continue_ckpt']
        self.torch_res18_ckpt = self.config['model']['torch_res18_ckpt']
        self.summary_freq = self.config['model']['summary_freq']
        self.auto_mask = self.config['model']['auto_mask']

        loader = DataLoader(trainable=True, **self.config)
        self.num_scales = loader.num_scales
        self.num_source = loader.num_source
        with tf.name_scope('data_loading'):
            tgt_image, src_image_stack, tgt_image_aug, src_image_stack_aug, intrinsics = loader.load_batch()
            tgt_image = tf.image.convert_image_dtype(tgt_image, dtype=tf.float32)
            src_image_stack = tf.image.convert_image_dtype(src_image_stack, dtype=tf.float32)
            tgt_image_aug = tf.image.convert_image_dtype(tgt_image_aug, dtype=tf.float32)
            src_image_stack_aug = tf.image.convert_image_dtype(src_image_stack_aug, dtype=tf.float32)
            if self.preprocess:
                tgt_image_net = self.preprocess_image(tgt_image_aug)
                src_image_stack_net = self.preprocess_image(src_image_stack_aug)
            else:
                tgt_image_net = tgt_image_aug
                src_image_stack_net = src_image_stack_aug

        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(True, **self.config)
            num_source = np.int(src_image_stack_net.get_shape().as_list()[-1] // 3)
            assert num_source == 2
            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)

            if self.pose_type == 'seperate':
                res18_ctp, _ = net_builder.build_resnet18(
                    tf.concat([tgt_image_net,src_image_stack_net[:, :, :, :3]], axis=3),
                    prefix='pose_'
                )
                res18_ctn, _ = net_builder.build_resnet18(
                    tf.concat([tgt_image_net, src_image_stack_net[:, :, :, 3:]], axis=3),
                    prefix='pose_'
                )
            elif self.pose_type == 'shared':
                res18_tp, _ = net_builder.build_resnet18(src_image_stack_net[:, :, :, :3])
                res18_tn, _ = net_builder.build_resnet18(src_image_stack_net[:, :, :, 3:])
                res18_ctp = tf.concat([res18_tc, res18_tp], axis=3)
                res18_ctn = tf.concat([res18_tc, res18_tn], axis=3)
            else:
                raise NotImplementedError

            pred_pose_ctp = net_builder.build_pose_net2(res18_ctp)
            pred_pose_ctn = net_builder.build_pose_net2(res18_ctn)

            pred_poses = tf.concat([pred_pose_ctp, pred_pose_ctn], axis=1)

            # res18_tp, _ = net_builder.build_resnet18(src_image_stack_net[:,:,:,:3])
            # res18_tn, _= net_builder.build_resnet18(src_image_stack_net[:,:,:,3:])
            #
            # pred_poses = net_builder.build_pose_net(res18_tp, res18_tc, res18_tn)

            pred_disp = net_builder.build_disp_net(res18_tc,skips_tc)

            H = tgt_image.get_shape().as_list()[1]
            W = tgt_image.get_shape().as_list()[2]


            pred_disp_rawscale = [tf.image.resize_bilinear(pred_disp[i], [loader.img_height, loader.img_width]) for i in range(self.num_scales)]
            pred_depth_rawscale = disp_to_depth(pred_disp_rawscale, self.min_depth, self.max_depth)

            tgt_image_pyramid = [tf.image.resize_nearest_neighbor(tgt_image, [np.int(H // (2 ** s)), np.int(W // (2 ** s))]) for s in range(self.num_scales)]

        with tf.name_scope('compute_loss'):
            tgt_image_stack_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            pixel_losses = 0.
            smooth_losses = 0.
            total_loss = 0.
            if self.auto_mask:
                pred_auto_masks1 = []
                pred_auto_masks2 = []
            for s in range(loader.num_scales):
                reprojection_losses = []
                for i in range(num_source):
                    curr_proj_image = projective_inverse_warp(src_image_stack[:,:,:, 3*i:3*(i+1)],
                                                              tf.squeeze(pred_depth_rawscale[s], axis=3),
                                                              pred_poses[:,i,:],
                                                              intrinsics=intrinsics[:,0,:,:])
                    curr_proj_error = tf.abs(curr_proj_image - tgt_image)

                    #if self.auto_mask and s == 0:
                        #proj_image_scale0.append(curr_proj_image)
                    #    proj_error_scale0.append(curr_proj_error)

                    reprojection_losses.append(self.compute_reprojection_loss(curr_proj_image, tgt_image))

                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                    else:
                        proj_image_stack = tf.concat([proj_image_stack,curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,curr_proj_error], axis=3)

                reprojection_losses = tf.concat(reprojection_losses, axis=3)

                combined = reprojection_losses
                if self.auto_mask:
                    identity_reprojection_losses = []
                    for i in range(num_source):
                        identity_reprojection_losses.append(self.compute_reprojection_loss(src_image_stack[:,:,:, 3*i:3*(i+1)], tgt_image))
                    identity_reprojection_losses = tf.concat(identity_reprojection_losses, axis=3)

                    identity_reprojection_losses += (tf.random_normal(identity_reprojection_losses.get_shape()) * 1e-5)

                    combined = tf.concat([identity_reprojection_losses, reprojection_losses], axis=3)

                    pred_auto_masks1.append(tf.expand_dims(tf.cast(tf.argmin(tf.concat([combined[:,:,:,1:2],combined[:,:,:,3:4]],axis=3), axis=3), tf.float32) * 255,-1))
                    pred_auto_masks2.append(tf.expand_dims(tf.cast(
                        tf.argmin(combined, axis=3) > 1,
                        tf.float32) * 255, -1))

                reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))



                pixel_losses += reprojection_loss

                smooth_loss = self.get_smooth_loss(pred_disp[s], tgt_image_pyramid[s]) / (2 ** s)
                smooth_losses += smooth_loss
                scale_total_loss = reprojection_loss + self.smoothness_ratio * smooth_loss
                total_loss += scale_total_loss

                tgt_image_stack_all.append(tgt_image)
                src_image_stack_all.append(src_image_stack_aug)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)

            total_loss /= loader.num_scales
            pixel_losses /= loader.num_scales
            smooth_losses /= loader.num_scales

        with tf.name_scope('train_op'):
            self.total_step = self.total_epoch * loader.steps_per_epoch
            self.global_step = tf.Variable(0,name='global_step',trainable=False)
            learning_rates = [self.start_learning_rate, self.start_learning_rate / 10 ]
            boundaries = [np.int(self.total_step * 3 / 4)]
            learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, learning_rates)
            optim = tf.train.AdamOptimizer(learning_rate, self.beta1)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.incr_global_step = tf.assign(self.global_step,self.global_step + 1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth_rawscale
        self.pred_disp = pred_disp
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_losses
        self.smooth_loss = smooth_losses
        self.tgt_image_all = tgt_image_stack_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        if self.auto_mask:
            self.pred_auto_masks1 = pred_auto_masks1
            self.pred_auto_masks2 = pred_auto_masks2

    def collect_summaries(self):
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.image('tgt_image',self.tgt_image_all[0])
        for s in range(self.num_scales):
            tf.summary.image('scale{}_disparity_color_image'.format(s), colorize(self.pred_disp[s], cmap='plasma'))
            tf.summary.image('scale{}_disparity_gray_image'.format(s), self.pred_disp[s])
            if self.auto_mask:
                tf.summary.image('scale{}_automask1_image'.format(s), self.pred_auto_masks1[s])
                tf.summary.image('scale{}_automask2_image'.format(s), self.pred_auto_masks2[s])
            for i in range(self.num_source):
                tf.summary.image('scale{}_projected_image_{}'.format (s, i),
                                 self.proj_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3])
                tf.summary.image('scale{}_proj_error_{}'.format(s, i),
                                     self.proj_error_stack_all[s][:, :, :, i * 3:(i + 1) * 3])
        tf.summary.histogram("tx", self.pred_poses[:, :, 0])
        tf.summary.histogram("ty", self.pred_poses[:, :, 1])
        tf.summary.histogram("tz", self.pred_poses[:, :, 2])
        tf.summary.histogram("rx", self.pred_poses[:, :, 3])
        tf.summary.histogram("ry", self.pred_poses[:, :, 4])
        tf.summary.histogram("rz", self.pred_poses[:, :, 5])

    def train(self, ckpt_dir):
        self.build_train()
        init = tf.global_variables_initializer()
        self.collect_summaries()

        # load weights from pytorch resnet 18 model
        if self.torch_res18_ckpt != '':
            assign_ops = load_resnet18_from_file(self.torch_res18_ckpt)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + [self.global_step],max_to_keep=10)
        sv = tf.train.Supervisor(logdir=ckpt_dir,save_summaries_secs=0,saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.model_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            sess.run(init)

            if self.continue_ckpt != '':
                print("Resume training from previous checkpoint: %s" % self.continue_ckpt)
                ckpt = tf.train.latest_checkpoint('{}/{}'.format(self.root_dir,self.continue_ckpt))
                self.saver.restore(sess, ckpt)

            elif self.torch_res18_ckpt != '':
                 sess.run(assign_ops)

            start_time = time.time()
            for step in range(1, self.total_step):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % self.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["pixel_loss"] = self.pixel_loss
                    fetches["smooth_loss"] = self.smooth_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % self.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [{}] | [{}/{}] | time: {:.4f} s/it | loss: {:.4f} pixel_loss: {:.4f} smooth_loss: {:.4f} ".format
                            (train_epoch, train_step, self.steps_per_epoch,
                                (time.time() - start_time)/self.summary_freq,
                             results["loss"], results["pixel_loss"], results["smooth_loss"]))
                    start_time = time.time()

                if step % (self.steps_per_epoch * 2) == 0:
                    self.save(sess, ckpt_dir, gs)
            self.save(sess, ckpt_dir, 'latest')

    def eval_depth(self, sess,ckpt_name):
        with open('data/kitti/test_files_eigen.txt', 'r') as f:
            test_files = f.readlines()
            test_files = [self.config['dataset']['root_dir'] + t[:-1] for t in test_files]
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        if not os.path.exists('{}/depth'.format(self.config['output_dir'])):
            os.makedirs('{}/depth'.format(self.config['output_dir']))
        basename = os.path.basename(ckpt_name)
        output_file = self.config['output_dir'] + '/depth/' + basename
        print('[MSG] save path: {}'.format(output_file))

        pred_all = []
        import cv2
        for t in range(len(test_files)):
            image = cv2.imread(test_files[t])

            image = cv2.resize(image,(self.loader.img_width, self.loader.img_height))
            print(test_files[t], np.array(image).shape)
            tgt_image_np = image

            tgt_image_np = np.expand_dims(tgt_image_np, axis=0)

            fetches = {
                'depth': self.pred_depth,
                'disp': self.pred_disp
            }

            results = sess.run(fetches, feed_dict={self.tgt_image: tgt_image_np})
            pred_depth = np.squeeze(results['depth'])
            pred_all.append(pred_depth)
            cv2.waitKey()


        np.save(output_file,pred_all)

    def eval_pose(self,sess, ckpt_name):
        raise NotImplementedError


    def eval(self, ckpt_name, eval_type):
        self.build_test(build_type=eval_type)

        self.saver = tf.train.Saver([var for var in tf.model_variables()], max_to_keep=10)

        for var in tf.model_variables():
            print(var)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if ckpt_name == '':
                print('No pretrained model provided, exit...')
                raise ValueError

            print('load pretrained model from: {}'.format(ckpt_name))
            latest_ckpt = tf.train.latest_checkpoint('{}'.format(ckpt_name))
            self.saver.restore(sess,latest_ckpt)
            if eval_type == 'depth':
                self.eval_depth(sess,ckpt_name)
            elif eval_type == 'pose':
                self.eval_pose(sess, ckpt_name)
            else:
                raise ValueError

    def test(self, ckpt_dir):
        self.build_test()

        self.saver = tf.train.Saver([var for var in tf.model_variables()], max_to_keep=10)

        for var in tf.model_variables():
            print(var)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            if ckpt_dir == '':
                print('No pretrained model provided, exit. ')
                raise ValueError

            print("load trained model")

            latest_ckpt = tf.train.latest_checkpoint('{}/{}'.format(self.root_dir,ckpt_dir))

            self.saver.restore(sess, latest_ckpt)

            file_list = self.loader.format_file_list(self.loader.dataset_dir, 'test')

            image_lists = file_list['image_file_list']

            import cv2

            for step in range(1, len(image_lists)):

                image = cv2.imread(image_lists[step])

                former_image_np = image[:,:self.loader.img_width,:]
                tgt_image_np = image[:,self.loader.img_width: self.loader.img_width*2,:]
                next_image_np = image[:,self.loader.img_width*2:,:]

                src_image_stack = np.concatenate((former_image_np, next_image_np), axis=2)

                tgt_image_np = np.expand_dims(tgt_image_np, axis=0)

                src_image_stack = np.expand_dims(src_image_stack, axis=0)

                fetches = {
                    'depth': self.pred_depth,
                    'poses':self.pred_poses
                }

                results = sess.run(fetches,feed_dict={self.tgt_image:tgt_image_np, self.src_image_stack: src_image_stack})

                disp_resized_np = np.squeeze(results['depth'])
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')

                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3][:,:,::-1] * 255).astype(np.uint8)

                disp_rgb = cv2.cvtColor(disp_resized_np,cv2.COLOR_GRAY2RGB)

                disp_rgb_int = (disp_rgb * 255.).astype(np.uint8)

                tgt_image_np = np.squeeze(tgt_image_np)

                toshow_image = np.vstack((tgt_image_np,colormapped_im, disp_rgb_int))

                print(toshow_image.shape)


                cv2.imshow('depth',toshow_image)
                cv2.waitKey(10)


    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}...".format(checkpoint_dir))
        if step == 'latest':
            self.saver.save(sess,os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)
