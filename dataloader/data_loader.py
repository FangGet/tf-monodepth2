from __future__ import division
import os
import random
import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(self, trainable=True, **config):
        self.config = config
        self.dataset_dir = self.config['dataset']['root_dir']
        self.batch_size = np.int(self.config['model']['batch_size']) 
        self.img_height = np.int(self.config['dataset']['image_height'])
        self.img_width = np.int(self.config['dataset']['image_width'])
        self.num_source = np.int(self.config['model']['num_source']) - 1
        self.num_scales = np.int(self.config['model']['num_scales'])

        self.trainable = trainable



    def load_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train' if self.trainable else 'val')
        image_paths_queue = tf.train.string_input_producer(file_list['image_file_list'],seed=seed,shuffle=True if self.trainable else False)
        cam_paths_queue = tf.train.string_input_producer(file_list['cam_file_list'],seed=seed,shuffle=True if self.trainable else False)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        # [H, W, 3] and [H, W, 3 * num_source]
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        # Form training batches
        src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics],
                               batch_size=self.batch_size)

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, image_all_aug = self.data_augmentation(
            image_all)

        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]

        tgt_image_aug = image_all_aug[:, :, :, :3]
        src_image_stack_aug = image_all_aug[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return tgt_image, src_image_stack, tgt_image_aug, src_image_stack_aug, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    # edit at 05/26 by Frank
    # add random brightness, contrast, saturation and hue to all source image
    # [H, W, (num_source + 1) * 3]
    def data_augmentation(self, im):
        def random_flip(im):
            def flip_one(sim):
                do_flip = tf.random_uniform([], 0, 1)
                return tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(sim), lambda: sim)

            im = tf.map_fn(lambda sim: flip_one(sim), im)
            #im = tf.cond(do_flip > 0.5, lambda: tf.map_fn(lambda sim: tf.image.flip_left_right(sim),im), lambda : im)
            return im

        def augment_image_properties(im):
            # random brightness
            brightness_seed = random.randint(0, 2**31 - 1)
            im = tf.image.random_brightness(im, 0.2, brightness_seed)

            contrast_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_contrast(im, 0.8, 1.2, contrast_seed)

            num_img = np.int(im.get_shape().as_list()[-1] // 3)

            # saturation_seed = random.randint(0, 2**31 - 1)
            saturation_im_list = []
            saturation_factor = random.uniform(0.8,1.2) #tf.random_ops.random_uniform([], 0.8, 1.2, seed=saturation_seed)
            for i in range(num_img):
                saturation_im_list.append(tf.image.adjust_saturation(im[:,:, 3*i: 3*(i+1)],saturation_factor))
                # tf.image.random_saturation(im[:,:, 3*i: 3*(i+1)], 0.8, 1.2, seed=saturation_seed))
            im = tf.concat(saturation_im_list, axis=2)

            #hue_seed = random.randint(0, 2 ** 31 - 1)
            hue_im_list = []
            hue_delta = random.uniform(-0.1,0.1) #tf.random_ops.random_uniform([], -0.1, 0.1, seed=hue_seed)
            for i in range(num_img):
                hue_im_list.append(tf.image.adjust_hue(im[:, :, 3 * i: 3 * (i + 1)],hue_delta))
                 #  tf.image.random_hue(im[:, :, 3 * i: 3 * (i + 1)], 0.1, seed=hue_seed))
            im = tf.concat(hue_im_list, axis=2)
            return im

        def random_augmentation(im):
            def augmentation_one(sim):
                do_aug = tf.random_uniform([], 0, 1)
                return tf.cond(do_aug > 0.5, lambda: augment_image_properties(sim), lambda : sim)
            im = tf.map_fn(lambda sim: augmentation_one(sim), im)
            #im = tf.cond(do_aug > 0.5, lambda: tf.map_fn(lambda sim: augment_image_properties(sim), im), lambda: im)
            return im

        im = random_flip(im)
        im_aug = random_augmentation(im)
        return im, im_aug

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        #print(image_seq.get_shape().as_list())
        tgt_start_idx = int(img_width * (num_source//2))
        # [h, w, 3]
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])

        # in case there are more images than 3
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
