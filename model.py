# Neeraj Prasad, based on Hyun Kim
# End-to-end one-shot autoencoder model
# Supports tensorboard, generates image data efficiently via 
# TFRecords. Saves best model w/ metadata

import os
import time
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
import load_data


class Autoencoder(object):
    """
    Autoencoder model class
    """
    def __init__(self, opt):
        self.opt = opt
        self.dataset = load_data.Dataset(opt)

        ## Repeatable Dataset for Training
        train_dataset = self.dataset.create_dataset(set_name='train', is_training=True)
        val_dataset = self.dataset.create_dataset(set_name='val', is_training=True)

        # Handles to switch datasets
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, train_dataset.output_types, train_dataset.output_shapes)

        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_one_shot_iterator()

        with tf.variable_scope("one_shot_ae", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.build()
            self.add_loss()

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        inc_global_step = tf.assign_add(global_step, 1, name='increment')

        grads = list(zip(gradients, params))
        for g, v in grads:
            gradient_summaries(g, v, opt)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updates = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

        # save network
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.opt.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def add_placeholders(self):
        """
        Adding placeholder
        """
        self.input_images, self.ans = self.iterator.get_next()
        self.preprocess()
        tf.summary.image('input', self.input_images)
        self.learning_rate = tf.placeholder(tf.float32, shape=())

    def build(self):
        """
        Build One-shot autoencoder
        """
        # Encoders
        self.encoder_1 = self.conv2d(self.input_images, filters=16, name="conv2_1")
        self.encoder_2 = self.conv2d(self.encoder_1, filters=32, name="conv2_2")
        self.encoder_3 = self.conv2d(self.encoder_2, filters=64, name="conv2_3")
        self.encoder_4 = self.conv2d(self.encoder_3, filters=128, name="conv2_4")

        shape = self.encoder_4.get_shape().as_list()

        dim = 1
        for d in shape[1:]:
            dim *= d
        
        # flatten
        self.latent = tf.reshape(self.encoder_4, [-1, dim], name="feature")

        # Decoders
        self.decoder_1 = self.conv2d_transpose(self.encoder_4, filters=64, name="conv2d_trans_1")
        self.decoder_2 = self.conv2d_transpose(self.decoder_1, filters=32, name="conv2d_trans_2")
        self.decoder_3 = self.conv2d_transpose(self.decoder_2, filters=16, name="conv2d_trans_3")
        self.decoder_4 = self.conv2d_transpose(self.decoder_3, filters=3, name="conv2d_trans_4")
        self.output_images = self.decoder_4

    def conv2d(self, bottom, filters, kernel_size=[5,5], stride=2, padding="SAME", name="conv2d"):
        layer = tf.layers.conv2d(bottom, filters, kernel_size, stride, padding)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        return layer        
    
    def conv2d_transpose(self, bottom, filters, kernel_size=[5,5], stride=2, padding="SAME", name="conv2d_trans"):
        layer = tf.layers.conv2d_transpose(bottom, filters, kernel_size, stride, padding)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        return layer

    def add_loss(self):
        """
        Defining a loss term (l2 loss)
        """
        with tf.variable_scope("loss"):
            diff = self.input_images - self.output_images
            self.loss = tf.divide(tf.nn.l2_loss(diff), tf.cast(tf.shape(diff)[0], dtype=tf.float32))
            tf.summary.scalar("loss", self.loss)

    def train_iter(self, session, train_writer, val_writer, iStep, iEpoch):
        """
        Train the network for one iteration
        """
        k = iStep*self.opt.batch_size + self.num_images_epoch*iEpoch
        feed_dict_train = {self.learning_rate:self.opt.learning_rate, self.handle:self.training_handle}
        feed_dict_val = {self.learning_rate:self.opt.learning_rate, self.handle:self.validation_handle}

        output_feed = [self.updates, self.summaries, self.global_step, self.loss]
        if iStep == 0:
            print("* epoch: " + str(float(k) / float(self.num_images_epoch)))
            [_, summaries, global_step, loss] = session.run(output_feed, feed_dict_train)
            train_writer.add_summary(summaries, k)
            print('train loss:', loss)
            sys.stdout.flush()

            [_, summaries, global_step, loss] = session.run(output_feed, feed_dict_val)
            val_writer.add_summary(summaries, k)
            print('val loss:', loss)
            sys.stdout.flush()
        else:
            session.run(output_feed, feed_dict_train)

        # Scheduling learning rate
        if int(global_step + 1) % self.opt.decay_every == 0:
            self.opt.learning_rate *= self.opt.decaying_rate

    def train(self, session):
        """
        Main training function
        """

        if not os.path.isfile(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/models/checkpoint'):
            session.run(tf.global_variables_initializer())
        elif opt.restart:
            print("RESTART")
            shutil.rmtree(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/models/')
            shutil.rmtree(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/train/')
            shutil.rmtree(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/val/')
            session.run(tf.global_variables_initializer())
        else:
            print("RESTORE")
            self.saver.restore(session, tf.train.latest_checkpoint(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/models/'))

        self.training_handle = session.run(train_iterator.string_handle())
        self.validation_handle = session.run(val_iterator.string_handle())

        self.summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/train', session.graph)
        val_writer = tf.summary.FileWriter(opt.precursor + opt.log_dir_base + opt.category + opt.name + '/val')
        print("STARTING EPOCH = ", session.run(global_step))

        parameters = tf.trainable_variables()
        num_parameters = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), parameters))
        print("Number of trainable parameters:", num_parameters)

        # For validation
        best_dev_loss = None
        epoch = 0

        self.num_images_epoch = len(self.dataset.train_addrs)
        for iEpoch in range(int(session.run(global_step)), opt.num_epochs):
            print('GLOBAL STEP:', session.run(global_step))
            epoch_start_time = time.time()
            self.saver.save(session, opt.precursor + opt.log_dir_base + opt.category + opt.name + '/models/model', global_step=iEpoch)

            for iStep in range(int(num_images_epoch/opt.batch_size)):
                # Epoch Counter
                iter_start_time = time.time()
                self.train_iter(session, train_writer, val_writer, iStep, iEpoch)
                iter_execution_time = time.time() - iter_start_time

            epoch_execution_time = time.time() - epoch_start_time
            print("Epoch:%d, execution time:%f" % (epoch, epoch_execution_time))
            sys.stdout.flush()
        session.run([inc_global_step])
        # save after finishing training epoch    
        self.bestmodel_saver.save(session, opt.precursor + opt.log_dir_base + opt.category + opt.name + '/models/bestmodel')
        train_writer.close()
        val_writer.close()
        print(':)')

    def preprocess(self):
        ims = tf.unstack(self.input_images, num=self.opt.batch_size, axis=0)
        process_imgs = []
        image_size = self.opt.image_size

        for image in ims:
            image = tf.random_crop(image, [image_size, image_size, 3])
            image = tf.image.per_image_standardization(image)*self.opt.scale
            process_imgs.append(image)

        self.input_images = tf.stack(process_imgs)



def gradient_summaries(grad, var, opt):

    if opt.extense_summary:
        tf.summary.scalar(var.name + '/gradient_mean', tf.norm(grad))
        tf.summary.scalar(var.name + '/gradient_max', tf.reduce_max(grad))
        tf.summary.scalar(var.name + '/gradient_min', tf.reduce_min(grad))
        tf.summary.histogram(var.name + '/gradient', grad)
