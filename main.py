# Copyright 2018 SAP LLC
import os
import time
import sys

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
import model, inst


def main():
    # load image names with full dir info and corresponding labels
    precursor = sys.argv[1]
    ID = int(sys.argv[2])
    factor = int(sys.argv[3])

    ID = ID + factor*1000
    
    opt = inst.gen_tune_exp(precursor)[ID]
    # start training from scratch
    if opt.mode == "train":
        new_model = model.Autoencoder(opt)
        with tf.Session() as sess:
            new_model.train(sess)
    elif opt.mode == "find_similar":
        search(opt)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time.time()-start_time)