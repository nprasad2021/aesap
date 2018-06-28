# Neeraj Prasad
# End-to-end one-shot autoencoder model
# Supports tensorboard, generates image data efficiently via 
# TFRecords. Saves best model w/ metadata
# Tests and generates plots for visual similarity comparison

import os
import time
import random
import shutil
import sys
import random

import numpy as np
import tensorflow as tf
import load_data, utils


class Autoencoder(object):
    """
    Autoencoder model class
    """
    def __init__(self, opt):

        self.opt = opt
        self.dataset = load_data.Dataset(opt)

        ## Repeatable Dataset for Training
        train_dataset = self.dataset.create_dataset(set_name='train')
        val_dataset = self.dataset.create_dataset(set_name='val')

        ## Non-repeatable Datasets for Testing
        generate_code_dataset = self.dataset.create_result_dataset(set_name='val', repeat=False)

        # Handles to switch datasets
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, train_dataset.output_types, train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_one_shot_iterator()
        self.val_iterator = val_dataset.make_one_shot_iterator()
        self.generate_code_iterator = generate_code_dataset.make_one_shot_iterator()

        with tf.variable_scope("one_shot_ae", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            if self.opt.build == 1: self.build_1()
            elif self.opt.build == 2: self.build_2()
            self.add_loss()

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)

        self.global_step = tf.Variable(0, name="global_step",  trainable=False)
        self.inc_global_step = tf.assign_add(self.global_step, 1, name='increment')

        grads = list(zip(gradients, params))
        for g, v in grads:
            utils.gradient_summaries(g, v, opt)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updates = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

        # save network
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.opt.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def add_placeholders(self):
        """
        Adding placeholder
        """
        self.input_images, self.ans, self.addrs = self.iterator.get_next()
        self.preprocess()
        tf.summary.image('input', self.input_images_1)
        self.learning_rate = tf.placeholder(tf.float32, shape=())

    def build_1(self):
        """
        Build One-shot autoencoder
        """
        # Encoders
        self.input_images_2 = tf.image.convert_image_dtype(self.input_images_1, tf.float32, saturate=True)
        self.encoder_1 = self.conv2d(self.input_images_1, filters=16, kernel_size=[7,7], name="conv2_1")
        self.encoder_2 = self.conv2d(self.encoder_1, filters=32, name="conv2_2")
        self.encoder_3 = self.conv2d(self.encoder_2, filters=64, name="conv2_3")
        self.encoder_final = self.conv2d(self.encoder_3, filters=128, name="conv2_4")

        self.lt_add()

        # Decoders
        self.decoder_1 = self.conv2d_transpose(self.encode_out, filters=64, name="conv2d_trans_1")
        self.decoder_2 = self.conv2d_transpose(self.decoder_1, filters=32, name="conv2d_trans_2")
        self.decoder_3 = self.conv2d_transpose(self.decoder_2, filters=16, name="conv2d_trans_3")
        self.decoder_4 = self.conv2d_transpose(self.decoder_3, filters=3, name="conv2d_trans_4")
        self.output_images = self.decoder_4
        tf.summary.image('output', self.output_images)
        self.output_images_1 = tf.image.convert_image_dtype(self.output_images, tf.float32, saturate=True)

    def build_2(self):
        """
        Build One-shot autoencoder
        """
        # Encoders
        self.input_images_2 = tf.image.convert_image_dtype(self.input_images_1, tf.float32, saturate=True)
        self.encoder_1 = self.conv2d(self.input_images_1, filters=16, kernel_size=[7,7], name="conv2_1")
        self.encoder_2 = self.conv2d(self.encoder_1, filters=32, name="conv2_2")
        self.encoder_final = self.conv2d(self.encoder_2, filters=64, name="conv2_3")


        self.lt_add()

        self.decoder_2 = self.conv2d_transpose(self.encode_out, filters=32, name="conv2d_trans_2")
        self.decoder_3 = self.conv2d_transpose(self.decoder_2, filters=16, name="conv2d_trans_3")
        self.decoder_4 = self.conv2d_transpose(self.decoder_3, filters=3, name="conv2d_trans_4")
        self.output_images = self.decoder_4
        tf.summary.image('output', self.output_images)
        self.output_images_1 = tf.image.convert_image_dtype(self.output_images, tf.float32, saturate=True)


    def conv2d(self, bottom, filters, kernel_size=None, stride=2, padding="SAME", name="conv2d"):
        if kernel_size is None:
            kernel_size = self.opt.kernel_size
        layer = tf.layers.conv2d(bottom, filters, kernel_size, stride, padding)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        return layer        
    
    def conv2d_transpose(self, bottom, filters, kernel_size=[3,3], stride=2, padding="SAME", name="conv2d_trans"):
        layer = tf.layers.conv2d_transpose(bottom, filters, kernel_size, stride, padding)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        return layer

    def vae(self):
        shape = self.encoder_final.get_shape().as_list()

        dim = 1
        for d in shape[1:]:
            dim *= d

        # flatten
        self.latent = tf.reshape(self.encoder_final, [-1, dim], name="feature")
        self.z_mu = tf.layers.dense(self.latent, self.opt.num_units)
        self.z_log_sigma_sq = tf.layers.dense(self.latent, self.opt.num_units)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        if self.opt.vae:
            self.encode_out = tf.reshape(tf.layers.dense(self.z, dim, tf.nn.relu), shape)
        else:
            self.encode_out = self.encoder_final

        self.class_1 = tf.layers.dense(self.latent, 200, tf.nn.relu)
        self.class_2 = tf.layers.dense(self.class_1, 50, tf.nn.relu)
        self.final_sm = tf.layers.dense(self.class_2, len(self.dataset.all_labels))

    def lt_add(self):
        shape = self.encoder_final.get_shape().as_list()

        dim = 1
        for d in shape[1:]:
            dim *= d

        # flatten
        self.latent = tf.reshape(self.encoder_final, [-1, dim], name="feature")
        self.encode_out = self.encoder_final

        self.class_1 = tf.layers.dense(self.latent, self.opt.num_units, tf.nn.relu)
        self.class_2 = tf.layers.dense(self.class_1, self.opt.num_units, tf.nn.relu)
        self.final_sm = tf.layers.dense(self.class_2, len(self.dataset.all_labels))


    def add_loss(self):
        """
        Defining a loss term (l2 loss)
        """
        with tf.variable_scope("loss"):
            
            #---------LATENT LOSS-------------------------
            # self.m_1 = tf.square(self.z_mu)
            # self.m_2 = tf.exp(self.z_log_sigma_sq)

            # self.m_3 = 1 + self.z_log_sigma_sq - self.m_1 - self.m_2

            # latent_loss = -0.5 * tf.reduce_sum(
            #     self.m_3, axis=1)
            # print(latent_loss.shape, 'latent_loss shape')
            # self.latent_loss = tf.reduce_mean(latent_loss)
            # print(self.latent_loss.shape, 'latent_loss shape after mean over batch')
            # tf.summary.scalar('latent loss', self.latent_loss)
            self.latent_loss = 0

            #--------RECONSTRUCTION LOSS------------------------
            # epsilon = 1e-10
            # recon_loss = -tf.reduce_sum(
            # self.input_images_1 * tf.log(epsilon+self.output_images) + (1-self.input_images_1) * tf.log(epsilon+1-self.output_images),
            #     axis=[1,2,3]) #Cross Entropy
            # print(recon_loss.shape, 'recon_loss shape')
            # self.recon_loss = tf.reduce_mean(recon_loss)
            # print(self.recon_loss.shape, 'recon_loss shape')
            # tf.summary.scalar('reconstruction loss', self.recon_loss)

            diff = (self.input_images_1 - self.output_images)/255
            if self.opt.loss == 'l2':
                self.recon_loss = tf.divide(tf.nn.l2_loss(diff), tf.cast(tf.shape(diff)[0], dtype=tf.float32))
            else: self.recon_loss = tf.divide(tf.reduce_sum(tf.abs(diff)), tf.cast(tf.shape(diff)[0], dtype=tf.float32))
            tf.summary.scalar('reconstruction loss', self.recon_loss)

            #----------CROSS ENTROPY LOSS------------------------
            
            self.accuracy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ans, logits=self.final_sm))
            tf.summary.scalar('cross_entropy', self.accuracy_loss)

            #----------TOTAL LOSS------------------
            self.loss = self.recon_loss + self.accuracy_loss
            tf.summary.scalar("total loss", self.loss)

            correct_prediction = tf.equal(tf.argmax(self.final_sm, 1), self.ans)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy', self.accuracy)

        # OLD METHOD OF L2 and L1 loss


    def train_iter(self, session, train_writer, val_writer, iStep, iEpoch):
        """
        Train the network for one iteration
        """
        k = iStep*self.opt.batch_size + self.num_images_epoch*iEpoch
        feed_dict_train = {self.learning_rate:self.opt.learning_rate, self.handle:self.training_handle}
        feed_dict_val = {self.learning_rate:self.opt.learning_rate, self.handle:self.validation_handle}

        output_feed_train = [self.updates, self.summaries, self.global_step, self.loss, self.recon_loss, self.accuracy_loss, self.accuracy]
        output_feed_val = [self.summaries, self.global_step, self.loss, self.accuracy]

        if iStep == 0:
            print("* epoch: " + str(float(k) / float(self.num_images_epoch)))
            [_, summaries, global_step, loss, reconl, accl, acc] = session.run(output_feed_train, feed_dict_train)

            train_writer.add_summary(summaries, k)
            print('train loss:', loss)
            print('reconstruction loss:', reconl)
            print('cross entropy loss:', accl)
            print('train acc:', acc)
            sys.stdout.flush()

            [summaries, global_step, loss, acc] = session.run(output_feed_val, feed_dict_val)
            val_writer.add_summary(summaries, k)

            if iEpoch == self.opt.num_epochs - 1:
                if not os.path.isfile(self.opt.precursor + self.opt.outfile):
                    open(self.opt.precursor + self.opt.outfile, 'a').close()
                with open(self.opt.precursor + self.opt.outfile, 'a+') as f:
                    f.write(str(self.opt.ID) + ',' + str(loss))
                    f.write('\n')

            print('val loss:', loss)
            print('val acc:', acc)
            sys.stdout.flush()
        else:
            [_, _, global_step, _, _, _, _] = session.run(output_feed_train, feed_dict_train)

        # Scheduling learning rate
        if int(global_step + 1) % self.opt.decay_every == 0:
            self.opt.learning_rate *= self.opt.decaying_rate

    def train(self, session):
        """
        Main training function
        """

        if not os.path.isfile(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/checkpoint'):
            session.run(tf.global_variables_initializer())
        elif self.opt.restart:
            print("RESTART")
            shutil.rmtree(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/')
            shutil.rmtree(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/train/')
            shutil.rmtree(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/val/')
            session.run(tf.global_variables_initializer())
        else:
            print("RESTORE")
            self.saver.restore(session, tf.train.latest_checkpoint(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/'))

        self.training_handle = session.run(self.train_iterator.string_handle())
        self.validation_handle = session.run(self.val_iterator.string_handle())

        self.summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/train', session.graph)
        val_writer = tf.summary.FileWriter(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/val')
        print("STARTING EPOCH = ", session.run(self.global_step))

        parameters = tf.trainable_variables()
        num_parameters = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), parameters))
        print("Number of trainable parameters:", num_parameters)

        # For validation
        best_dev_loss = None
        epoch = 0

        self.num_images_epoch = len(self.dataset.train_addrs)
        for iEpoch in range(self.opt.num_epochs):
            print('GLOBAL STEP:', session.run(self.global_step))
            epoch_start_time = time.time()
            self.saver.save(session, self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/model', global_step=iEpoch)

            for iStep in range(int(self.num_images_epoch/self.opt.batch_size)):
                # Epoch Counter
                iter_start_time = time.time()
                self.train_iter(session, train_writer, val_writer, iStep, iEpoch)
                iter_execution_time = time.time() - iter_start_time

            epoch_execution_time = time.time() - epoch_start_time
            print("Epoch:%d, execution time:%f" % (epoch, epoch_execution_time))
            sys.stdout.flush()
        session.run([self.inc_global_step])
        # save after finishing training epoch    
        self.bestmodel_saver.save(session, self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/bestmodel/')
        train_writer.close()
        val_writer.close()

        print('Successfully completed training')
        print(':)')


    def tester(self, session):
        print('Enter Testing')
        if not os.path.isfile(self.opt.pipeline + '/models/checkpoint'):
            print("MODEL NOT TRAINED. RETRAIN IMMEDIATELY")
            return
        else:
            print("RESTORE")
            self.saver.restore(session, tf.train.latest_checkpoint(self.opt.precursor + self.opt.log_dir_base + self.opt.category + self.opt.name + '/models/'))
        sys.stdout.flush()


        gen_code_handle = session.run(self.generate_code_iterator.string_handle())
        feed_dict_gen_code = {self.learning_rate:self.opt.learning_rate, self.handle:gen_code_handle}
        start = True
        num_iter_true = 0
        latent_code, addrs_code, label = session.run([self.latent, self.addrs, self.ans], feed_dict_gen_code)
        
        while True:
            try:
                latent_code, addrs_code, label = session.run([self.latent, self.addrs, self.ans], feed_dict_gen_code)
                if start:
                    print('small latent code shape', latent_code.shape)
                    latent_codes_ref = latent_code
                    addrs_codes_ref = addrs_code
                    labels_ref = label
                    start = False
                else:
                    latent_codes_ref = np.concatenate(latent_codes_ref, latent_code, axis=0)
                    print(latent_codes_ref.shape, 'iteration shape')
                    addrs_codes_ref = np.concatenate(addrs_codes_ref, addrs_code, axis=0)
                    labels_ref = np.concatenate(labels_ref, label, axis=0)
                num_iter_true += 1
            except:
                print('dataset iteration finished')
                break
        print('num_iter_true', num_iter_true)
        print('latent codes ref shape:', latent_codes_ref.shape)
        print('latent code initial shape', latent_code.shape)

        validation_handle = session.run(self.val_iterator.string_handle())
        num_iter = len(self.dataset.val_addrs)//self.opt.batch_size
        feed_dict_test = {self.learning_rate:self.opt.learning_rate, self.handle:validation_handle}
        output_feed = [self.latent, self.input_images_2, self.output_images_1, self.ans, self.addrs, self.accuracy]

        top_k = 0
        total_acc = 0

        for mini in range(num_iter):

            latent_query, input_im, output_im, label_query, addr_query, acc = session.run(output_feed, feed_dict_test)
            print('latent_query shape', latent_query.shape)
            utils.autovis(mini, input_im, output_im, self.opt.batch_size, self.opt.figline)
            utils.simrank(mini, latent_query, latent_codes_ref, addr_query, addrs_codes_ref, self.opt)
            top_k += utils.top_k_score(latent_query, latent_codes_ref, label_query, labels_ref, self.opt.K, self.opt.similarity_distance)

            total_acc += acc

        top_k_avg = top_k / num_iter
        report_acc = total_acc/num_iter

        if not os.path.isfile(self.opt.resultline):
            open(self.opt.resultline, 'a').close()
        with open(self.opt.resultline, 'a+') as f:
            f.write(str(self.opt.ID) + ',' + str(top_k_avg))
            f.write('\n')

        if not os.path.isfile(self.opt.accline):
            open(self.opt.accline, 'a').close()
        with open(self.opt.accline, 'a+') as f:
            f.write(str(self.opt.ID) + ',' + str(report_acc))
            f.write('\n')

        print('Top K:', score)
        print('Final Accuracy:', report_acc)
        print('Successfully completed testing :)')

    def preprocess(self):
        ims = tf.unstack(self.input_images, num=self.opt.batch_size, axis=0)
        process_imgs = []
        image_size = self.opt.image_size

        for image in ims:
            image = tf.random_crop(image, [image_size, image_size, 3])
            image = tf.image.per_image_standardization(image)*self.opt.scale + self.opt.slide
            process_imgs.append(image)

        self.input_images_1 = tf.stack(process_imgs)

    def deprocess(self, images, mean, stdev, norm):
        print(mean.shape, 'mean shape')
        print(stdev.shape, 'stdev shape')

        ims = np.split(images, self.opt.batch_size)
        stdev = np.split(stdev, self.opt.batch_size)
        mean = np.split(mean, self.opt.batch_size)

        print(ims[0].shape, 'images shape in deprocess')

        process_imgs = []

        for image, mn, std in zip(ims, mean, stdev):
            if norm:
                std = max(std, 1.0/np.sqrt(image.size))
                image = ((image - self.opt.slide) / self.opt.scale)*std + mn
            if np.amin(image) < 0: image = image + 127
            image = 255*image/np.amax(image)
            image.astype(int)
            process_imgs.append(image)

        return np.concatenate(process_imgs)