# Neeraj Prasad
# Encode JPEG image data into TFRecords w/ labels
# Create dataset object automatically load,
# process, and return batches of images from TFRecords

from random import shuffle
import glob
import os
import sys
import random

import tensorflow as tf
import numpy as np
import parameters


class Dataset:

	def __init__(self, opt):
		self.opt = opt    # parameters
		self.tdata = self.opt.tdata
		self.num_threads = 8

		### Write TF Records
		self.write()

	def load_data(self, shuffle_data=True):

		self.train_addrs, self.train_labels = self.reformat_addrs(self.opt.train_dir, shuffle_data)
		self.val_addrs, self.val_labels = self.reformat_addrs(self.opt.val_dir, shuffle_data)

	def reformat_addrs(self, path, shuffle_data):
		addrs = []
		labels = []

		all_labels = os.listdir(path)
		all_labels.sort()

		for ind, lbl in enumerate(all_labels):

			lbl_addrs = glob.glob(path + lbl + '/*.jpg')
			labels += [ind]*len(lbl_addrs)
			addrs += lbl_addrs

		if shuffle_data:

			c = list(zip(addrs, labels))
			shuffle(c)
			addrs, labels = zip(*c)
		return addrs, labels

	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def write_TFRecords(self):

		if not os.path.isdir(self.opt.tfr_out):
			os.makedirs(self.opt.tfr_out)
		for trval in ['train', 'val']:

			if trval == 'train':
				addrs = self.train_addrs
				labels = self.train_labels
			else:
				addrs = self.val_addrs
				labels = self.val_labels

			write_filename = self.opt.tfr_out + trval + '.tfrecords'
			if os.path.exists(write_filename) and self.opt.reuse_TFRecords:
				continue

			writer = tf.python_io.TFRecordWriter(write_filename)

			tfcoder = ImageCoder()
			for i in range(len(addrs)):
				if i % 1000 == 0:
					print(trval + ' data: {}/{}'.format(i, len(addrs)))
					sys.stdout.flush()

				img, height, width = _process_image(addrs[i],tfcoder)
				lbl = labels[i]

				feature = {trval + '/label': self._int64_feature(labels[i]),
					trval + '/image': self._bytes_feature(img),
					trval + '/width': self._int64_feature(width),
					trval + '/height': self._int64_feature(height)}
				example = tf.train.Example(features=tf.train.Features(feature=feature))
				writer.write(example.SerializeToString())
			writer.close()
			sys.stdout.flush()

	def write(self):
		self.load_data()
		self.write_TFRecords()

	##### NON-REPEATABLE DATASET
	def create_dataset(self, set_name, repeat=True):

	    image_size = tf.cast(self.opt.image_size, tf.int32)
	    # Transforms a scalar string `example_proto` into a pair of a scalar string and
	    # a scalar integer, representing an image and its label, respectively.
	    def _parse_function(example_proto):
	    	features = {set_name + '/label': tf.FixedLenFeature((), tf.int64, default_value=1),
	    		set_name + '/image': tf.FixedLenFeature((), tf.string, default_value=""),
	    		set_name + '/height': tf.FixedLenFeature([], tf.int64),
	    		set_name + '/width': tf.FixedLenFeature([], tf.int64)}

	    	parsed_features = tf.parse_single_example(example_proto, features)
	    	image = tf.image.decode_jpeg(parsed_features[set_name + '/image'],channels=3)
	    	image = tf.cast(image, tf.float32)

	    	S = tf.stack([tf.cast(parsed_features[set_name + '/height'], tf.int32),
	    		tf.cast(parsed_features[set_name + '/width'], tf.int32), 3])
	    	image = tf.reshape(image, S)
	    	image = tf.image.resize_images(image, [image_size, image_size])
	    	
	    	return image, parsed_features[set_name + '/label']

	    tfrecords_path = self.opt.tfr_out

	    filenames = [tfrecords_path + set_name + '.tfrecords']
	    
	    dataset = tf.data.TFRecordDataset(filenames)
	    dataset = dataset.map(_parse_function)
	    if repeat:
	    	dataset = dataset.repeat()  # Repeat the input indefinitely.
	    return dataset.batch(self.opt.batch_size)

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._jpeg_data, channels=3)

        # Initializes function that encodes RGB JPEG data.
        self._image = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
        self._image, format='rgb', quality=100)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        return self._sess.run(self._encode_jpeg,
                          feed_dict={self._image: image})


def _process_image(filename, coder):
    """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB properly.
    assert len(image.shape) == 3
    assert image.shape[2] == 3


    height = image.shape[0]
    width = image.shape[1]

    image_buffer = coder.encode_jpeg(image)

    return image_buffer, height, width

if __name__ == "__main__":

	### Test: tfrecords files for desired dataset load
	### write_TFRecords()
	opt = parameters.Experiment()
	aesap = Dataset(opt)
