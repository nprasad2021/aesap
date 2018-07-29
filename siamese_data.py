# Neeraj Prasad - (created for Siamese Dataset)
# Inherits from Class Dataset
# Encode JPEG image data into TFRecords w/ labels
# Create dataset object automatically load,
# process, and return batches of images from TFRecords


####### HOW TO IMPLEMENT DATASET ########
# To instantiate your dataset object:
# dataset = SiameseDataset(FLAGS)

# FLAGS must have the following variables:
#	train_dir (training directory)
#	val_dir (validation directory)
#	tfr_out (where to place TFRecords file)
#	reuse_TFRecords (whether to use existing TFRecords, or to remake them)
#	image_size (desired image size)
#	batch_size (desired batch_size)
#	tfr_eval (address to TFRrecords for testing)

# REQUIREMENTS:
#	1) Must have load_data.py (SiameseDataest inherits from Dataset)
#	2) Must have utils.py (for image processing utilities)
#	3) Must use dataset iterator functionality when loading images

################## END ####################

from random import shuffle
import glob
import os
import sys
import random

import tensorflow as tf
import numpy as np
import parameters, utils

import feature_generator_siamese

class SiameseDataset(Dataset):

	def load_data(self, shuffle_data=True):
		pass

	def write_TFRecords(self):

		if not os.path.isdir(self.opt.tfr_out):
			os.makedirs(self.opt.tfr_out)
		for trval in ['train', 'val']:

			if trval == 'train':
				data_dir = self.opt.train_dir
			else:
				data_dir = self.opt.val_dir

			features = feature_generator_siamese.Feature(data_dir)

			write_filename = self.opt.tfr_out + trval + '.tfrecords'
			if os.path.exists(write_filename) and self.opt.reuse_TFRecords:
				continue

			writer = tf.python_io.TFRecordWriter(write_filename)

			tfcoder = utils.ImageCoder()
			for i, feature in enumerate(features):
				if i % 1000 == 0:
					print(trval + ' data: {}/{}'.format(i, 1))
					sys.stdout.flush()

				img_left, height_left, width_left = utils._process_image(feature[0],tfcoder)
				img_right, height_right, width_right = utils._process_image(feature[1],tfcoder)

				feature_record = {trval + '/label': self._int64_feature(int(feature[4])),
					trval + '/image_left': self._bytes_feature(img_left),
					trval + '/image_right': self._bytes_feature(img_right),
					trval + '/width_left': self._int64_feature(width_left),
					trval + '/height_left': self._int64_feature(height_left),
					trval + '/width_right': self._int64_feature(width_right),
					trval + '/height_right': self._int64_feature(height_right),
					trval + '/addr_left': self._bytes_feature(feature[0].encode()),
					trval + '/addr_right': self._bytes_feature(feature[1].encode())}

				example = tf.train.Example(features=tf.train.Features(feature=feature_record))
				writer.write(example.SerializeToString())

			writer.close()
			sys.stdout.flush()

	##### NON-REPEATABLE DATASET
	def create_dataset(self, set_name, repeat=True):

	    image_size = tf.cast(self.opt.image_size, tf.int32)
	    # Transforms a scalar string `example_proto` into a pair of a scalar string and
	    # a scalar integer, representing an image and its label, respectively.
	    def _parse_function(example_proto):
	    	features = {set_name + '/label': tf.FixedLenFeature((), tf.int64, default_value=1),
	    		set_name + '/image_left': tf.FixedLenFeature((), tf.string, default_value=""),
	    		set_name + '/image_right': tf.FixedLenFeature((), tf.string, default_value=""),
	    		set_name + '/height_left': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/width_left': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/height_right': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/width_right': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/addr_left': tf.FixedLenFeature([], tf.string, default_value=""),
	    		set_name + '/addr_right': tf.FixedLenFeature([], tf.string, default_value="")}

	    	parsed_features = tf.parse_single_example(example_proto, features)
	    	image_left = tf.image.decode_jpeg(parsed_features[set_name + '/image_left'],channels=3)
	    	image_left = tf.cast(image_left, tf.float32)

	    	S = tf.stack([tf.cast(parsed_features[set_name + '/height_left'], tf.int32),
	    		tf.cast(parsed_features[set_name + '/width_left'], tf.int32), 3])
	    	image_left = tf.reshape(image_left, S)
	    	image_left = tf.image.resize_images(image_left, [image_size, image_size])


	    	image_right = tf.image.decode_jpeg(parsed_features[set_name + '/image_right'],channels=3)
	    	image_right = tf.cast(image_right, tf.float32)

	    	S = tf.stack([tf.cast(parsed_features[set_name + '/height_right'], tf.int32),
	    		tf.cast(parsed_features[set_name + '/width_right'], tf.int32), 3])
	    	image_right= tf.reshape(image_right, S)
	    	image_right = tf.image.resize_images(image_right, [image_size, image_size])
	    	
	    	return image_left, image_right, parsed_features[set_name + '/label'], parsed_features[set_name + '/addr_left'].decode(), parsed_features[set_name + '/addr_right'].decode()

	    tfrecords_path = self.opt.tfr_out
	    filenames = [tfrecords_path + set_name + '.tfrecords']

	    dataset = tf.data.TFRecordDataset(filenames)
	    dataset = dataset.map(_parse_function)
	    if repeat:
	    	dataset = dataset.repeat()
	    return dataset.batch(self.opt.batch_size)

	def create_result_dataset(self, set_name, repeat=False):
	    image_size = tf.cast(self.opt.image_size, tf.int32)
	    # Transforms a scalar string `example_proto` into a pair of a scalar string and
	    # a scalar integer, representing an image and its label, respectively.
	    def _parse_function(example_proto):
	    	features = {set_name + '/label': tf.FixedLenFeature((), tf.int64, default_value=1),
	    		set_name + '/image_left': tf.FixedLenFeature((), tf.string, default_value=""),
	    		set_name + '/image_right': tf.FixedLenFeature((), tf.string, default_value=""),
	    		set_name + '/height_left': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/width_left': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/height_right': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/width_right': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/addr_left': tf.FixedLenFeature([], tf.string, default_value=""),
	    		set_name + '/addr_right': tf.FixedLenFeature([], tf.string, default_value="")}

	    	parsed_features = tf.parse_single_example(example_proto, features)
	    	image_left = tf.image.decode_jpeg(parsed_features[set_name + '/image_left'],channels=3)
	    	image_left = tf.cast(image_left, tf.float32)

	    	S = tf.stack([tf.cast(parsed_features[set_name + '/height_left'], tf.int32),
	    		tf.cast(parsed_features[set_name + '/width_left'], tf.int32), 3])
	    	image_left = tf.reshape(image_left, S)
	    	image_left = tf.image.resize_images(image_left, [image_size, image_size])


	    	image_right = tf.image.decode_jpeg(parsed_features[set_name + '/image_right'],channels=3)
	    	image_right = tf.cast(image_right, tf.float32)

	    	S = tf.stack([tf.cast(parsed_features[set_name + '/height_right'], tf.int32),
	    		tf.cast(parsed_features[set_name + '/width_right'], tf.int32), 3])
	    	image_right= tf.reshape(image_right, S)
	    	image_right = tf.image.resize_images(image_right, [image_size, image_size])
	    	
	    	return image_left, image_right, parsed_features[set_name + '/label'], parsed_features[set_name + '/addr_left'].decode(), parsed_features[set_name + '/addr_right'].decode()

	    tfrecords_path = self.opt.tfr_eval
	    filenames = [tfrecords_path + set_name + '.tfrecords']

	    dataset = tf.data.TFRecordDataset(filenames)
	    dataset = dataset.map(_parse_function)

	    if repeat:
	    	dataset = dataset.repeat()

	    return dataset.batch(self.opt.batch_size)



if __name__ == "__main__":

	### Test: tfrecords files for desired dataset load
	### write_TFRecords()
	precursor = sys.argv[1]
	for dataset in ['sap_data', 'all_data', 'open_data']:
		rewrite_TFRecords(dataset=dataset, precursor=precursor)