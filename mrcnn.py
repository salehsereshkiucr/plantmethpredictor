import tensorflow.compat.v1 as tf
import profile_generator as pg
import pandas as pd
import preprocess as preprocess
import random
from sklearn.metrics import accuracy_score
import numpy as np

keep_prob = 0.5

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def net_MRCNN(x_fs):
    W_conv1 = weight_variable([1, 4, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.conv2d(x_fs, W_conv1, strides=[1, 1, 4, 1], padding='VALID') + b_conv1
    h_conv1 = tf.reshape(h_conv1, [-1, 20, 20, 16])
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    W_conv3 = weight_variable([3, 3, 32, 48])
    b_conv3 = bias_variable([48])
    h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='VALID')+ b_conv3
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1,1,1,1], padding='VALID')+ b_conv4
    W_fc1 = weight_variable([2*2*64, 80])
    b_fc1 = bias_variable([80])
    h_pool4 = tf.reshape(h_conv4, [-1, 2*2*64])
    h_fc1 = tf.matmul(h_pool4, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([80, 1])
    b_fc2 = bias_variable([1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def data_preprocess(X, Y):
    X = np.delete(X, range(4, X.shape[2]), 2)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    Y = np.expand_dims(Y, axis=1)
    X = X.astype('float32')
    Y = Y.astype('float32')
    return X, Y

def test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic):
    methylated, unmethylated = preprocess.methylations_subseter(methylations_test, window_size)
    test_sample_size = int(min(50000, 2*len(methylated), 2*len(unmethylated)))
    test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
    random.shuffle(test_sample_set)
    test_profiles, test_targets = pg.get_profiles(methylations_test, test_sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
    x_test, y_test = data_preprocess(test_profiles, test_targets)
    return x_test, y_test

def run_experiment(cnfg, context, coverage_threshold = 10, data_size=500000):

    window_size = 400
    batch_size = 10
    organism_name = cnfg['organism_name']
    sequences_onehot, methylations, annot_seqs_onehot, num_to_chr_dic = pg.get_processed_data(cnfg, context, coverage_threshold=coverage_threshold)
    methylations_train, methylations_test = preprocess.seperate_methylations(organism_name, methylations, from_file=False)
    methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, window_size)

    tf.disable_eager_execution()
    tf_train_dataset_ph = tf.placeholder(tf.float32, shape=(None, 400, 4, 1), name='X')
    tf_train_labels_ph = tf.placeholder(tf.float32, shape=(None, 1), name='Y')
    logits = net_MRCNN(tf_train_dataset_ph)
    loss = tf.reduce_mean(tf.square(tf_train_labels_ph - logits), reduction_indices=[1])
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    x_test, y_test = test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic)
    test_prediction = net_MRCNN(x_test)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for chunk in range(0, data_size, batch_size):
            if chunk+batch_size > data_size:
                break
            else:
                sample_set = methylated_train[chunk:chunk+batch_size]+unmethylated_train[chunk:chunk+batch_size]
            random.shuffle(sample_set)
            profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
            X, Y = data_preprocess(profiles, targets)
            feed_dict = {tf_train_dataset_ph : X, tf_train_labels_ph : Y}
            sess.run(optimizer, feed_dict=feed_dict)
        y_pred = test_prediction.eval()
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return accuracy_score(y_pred, y_test)















