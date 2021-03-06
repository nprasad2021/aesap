# Neeraj Prasad
# Experiment Specification
# Run train and/or test

import os
import time
import sys

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
import model, inst


def main():
    
    # Load Experiment Specifications
    precursor = sys.argv[1]
    ID = int(sys.argv[2])
    factor = int(sys.argv[3])

    ID += factor*1000

    opt = inst.gen_tune_exp(precursor)[ID]
    opt.reinitialize_paths()

    print(opt)

    #TRAIN
    tf.reset_default_graph()
    if opt.mode == "train" or opt.mode == 'both':
        train_model = model.Autoencoder(opt)
        with tf.Session() as sess:
            train_model.train(sess)

    #TEST
    tf.reset_default_graph()
    if opt.mode == "test" or opt.mode == 'both':
        test_model = model.Autoencoder(opt)
        with tf.Session() as sess:
            test_model.tester(sess)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time.time()-start_time)