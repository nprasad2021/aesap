# Unsupervised Transfer Learning with Autoencoders in Tensorflow

Neeraj Prasad

## Background (Variational Autoencoders)

An autoencoder is a type of machine learning model that can learn to extract a robust, space-efficient feature vector from an image. The model has two parts - an encoder (e) and a decoder (p). The encoder takes as an input the image, and outputs a low-dimensional feature vector. The decoder takes as an input this low-dimensional feature vector, and recreates the original shape. 

The model is trained to minimize the distance between the input image and the output image, such that the model is able to recreate the input image. The distance metric used varies - here l1, l2, and cosine distance were tested. L1 distance performed the best.

The low-dimensional feature vector, which serves at the output to the encoder and the input to the decoder, is known as the latent variable. A 224x224x3 input image (containing a total 150,528 features) can be reduce to a latent vector of shape 4096x1 (4096 features). Since this latent vector can be used to recreate the input, it efficiently encapsulates information specifically relevant to the object in the image. 

In a deep autoencoder, a series of convolutional layers (the encoder) reduces the input image size to a latent variable. A series of deconvolutional layers (the decoder) maps the latent variable to an output image. The deep autoencoder is trained to recreate the input image in the output.

While the autoencoder model provides a robust unsupervised way to encode an image, it is prone to overfitting. The model can ‘memorize’ a mapping from latent variable to image, thus losing its ability to encode only valuable object-specific information in the encoding. I use a variational autoencoder, which fits the latent vector to a unit gaussian before decoding, enforcing normalization.

## Technical Implementation

The follow section details the technical implementation of the variational autoencoder.

### Parameters

All parameters are described in the class Experiment defined in '<parameters.py>'.
To tune hyperparameters a number of different variations are described in '<inst.py>'.


### Input Function

A private database of 140,000 images were used to train this model. Due to the high-dimensionality of image data, and due to the size of the dataset, an efficient input pipeline is necessary. 

Opening and closing image files repeatedly cost the most time every iteration - thus all image data was stored in a TFRecords format. This is described in the file '<dataset.py>'.

Image data was read with the tensorflow's dataset API. Shuffling and other data processing functions were applied to images before being fed into '<model.py>'

### Model

Four convolutional layers are applied to the input, together forming the encoder. Each layer consists of a convolutional transformation, batch normalization, and ReLu activation. 

The output to the encoder is fit to the unit gaussian, the defining motif of the variational autoencoder. The mean and variance from the unit gaussian are transformed into an input for the decoder.

Four deconvolutional layers, each associated with batch normalizatin and ReLu, form the decoder. The distance between the output of the decoder and the input image is measure by some distance metric. The model is trained to minimize this reconstruction loss.

A classification cross-entropy loss is optionally imposed on the sigmoidal output of a fully connected layer to the latent variable.

## Evaluation

When the vanilla autoencoder model was trained only to minimize reconstruction loss, the autoencoder performed extremely well, with image outputs approximating inputs almost exactly. The variational autoencoder also performed well on the reconstruction task (although average distance between inputs and outputs exceeded that of the vanilla autoencoder). Best results were consistently achieved with use of L1 Distance metric.

When the model was trained for both reconstruction loss and cross-entropy classification loss, the model's reconstruction distance increased significantly as expected. The model did not perform especially well on the classification task (65% for 10 classes).

