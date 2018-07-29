### Created by @neeraj
### Various Image Processing and 
### plot creation utilities


import os
import random
import scipy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import load_data


def autovis(id_num, inputim, outputim, batch_size, figline):
    inims = np.squeeze(np.split(inputim, batch_size))
    outims = np.squeeze(np.split(outputim, batch_size))

    print(inims[0].shape, 'input images shape')
    print(outims[0].shape, 'output images shape')

    fig = plt.figure()
    numofpairs = 1
    frame1 = plt.gca()
    for original, modified in zip(inims, outims):

        if numofpairs == 21:
            break
        ax = fig.add_subplot(5,4,numofpairs)
        ax.imshow(original)
        numofpairs += 1
        ax = fig.add_subplot(5,4,numofpairs)
        ax.imshow(modified)
        numofpairs += 1

    if not os.path.exists(figline + 'autovis/'):
        os.makedirs(figline + 'autovis/')

    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.axis('off')
    plt.savefig(figline + 'autovis/' + str(id_num) + '.pdf', dpi=1000)
    plt.close()

def simrank(id_num, lat_batch, cod, addrs_codes, addr, opt):
    distance = {'cosine':cos_similarity, 'l2':l2_similarity}
    latvecs = np.squeeze(np.split(lat_batch, opt.batch_size))
    codes = np.squeeze(np.split(cod, cod.shape[0]))

    search_result = np.ones(((opt.image_size * min(opt.display_num, opt.batch_size)), opt.image_size * (opt.K + 1), 3), dtype=np.uint8) * 255

    print(latvecs[0].shape, 'latent vector shape')
    print(codes[0].shape, 'codes shape')

    dim, k = min(opt.display_num, opt.batch_size), opt.K
    for i in range(dim):
        print(addr[i])
        knn = knn_search(latvecs[i], codes, k, addrs_codes, distance[opt.similarity_distance])
        for j in range(k+1):

            if j == 0: ref_image = scipy.misc.imread(addr[i].decode())
            else: ref_image = scipy.misc.imread(knn[j-1].decode())
            search_result[opt.image_size*i:opt.image_size*(i+1), j*opt.image_size:(j+1)*opt.image_size, :] = scipy.misc.imresize(ref_image, (opt.image_size, opt.image_size))

    if not os.path.exists(opt.figline + 'simrank/'):
        os.makedirs(opt.figline + 'simrank/')

    scipy.misc.imsave(opt.figline + 'simrank/' + str(id_num) + '.pdf', search_result)


def top_k_score(latent_query, latent_reference, labels_query, labels_reference, K, similarity_metric):
    distance = {'cosine':cos_similarity, 'l2':l2_similarity}

    lq = np.squeeze(np.split(latent_query, latent_query.shape[0]))
    lr = np.squeeze(np.split(latent_reference, latent_reference.shape[0]))

    bq = np.squeeze(np.split(labels_query, labels_query.shape[0]))
    br = np.squeeze(np.split(labels_reference, labels_reference.shape[0]))

    correct = 0
    total = 0

    for latent, label in zip(lq, bq):
        knn = knn_search(latent, lr, K, br, distance[similarity_metric])
        for i in knn:
            total += 1
            if i == label:
                correct += 1

    return correct/total


def knn_search(x, D, K, ims, sim):
    '''
    KNN search algorithm.
    Sort images in order by most similar
    according to Manhattan distance
    '''
    new_array = []
    final = []
    for item in D:
        new_array.append(sim(x, item))
    ind_array = np.argsort(new_array)
    for item in ind_array:
        final.append(ims[item])
    return final[-1*K:]

def cos_similarity(a, b):
    '''
    More representative of 
    similarity in high dimensional spaces
    '''
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def l2_similarity(a, b):
    return np.sum(np.square(np.subtract(a,b)))

def gradient_summaries(grad, var, opt):

    if opt.extense_summary:
        tf.summary.scalar(var.name + '/gradient_mean', tf.norm(grad))
        tf.summary.scalar(var.name + '/gradient_max', tf.reduce_max(grad))
        tf.summary.scalar(var.name + '/gradient_min', tf.reduce_min(grad))
        tf.summary.histogram(var.name + '/gradient', grad)

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
