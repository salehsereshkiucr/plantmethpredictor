import gc
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import configs as configs


num_steps = 200
batch_size = 128
display_step = 10




def load_data(root, organism_name, context, i, mode):
    X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
    X = np.delete(X, range(4,X.shape[2]), 2)
    b = [j for j in range(300)] + [j for j in range(700, 1000)]
    X = np.delete(X, b, 1)
    Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size), Y] = 1
    Y = b
    X = X.reshape(list(X.shape) + [1])
    X = np.swapaxes(X, 1, 2)

    PROFILE_COLS = X.shape[1]
    PROFILE_ROWS = X.shape[2]

    test_percent = 0.2
    test_val_percent = 0.5
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)
    return x_train, x_test, y_train, y_test

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
    print (h_conv1)

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    print (h_pool2)

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
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
    print (h_fc1_drop)

    W_fc2  = weight_variable([80, 1])
    b_fc2  = bias_variable([1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print (y_conv)

    return y_conv

tf.disable_eager_execution()

root = '/home/csgrads/ssere004/output_cpgenieannot/'
organism_name = 'Arabidopsis'
context = 'CG'
i = 1
mode = 'Seq'
for at in configs.Arabidopsis_config['annot_types']:
    mode+=at

x_train, x_test, y_train, y_test = load_data(root, organism_name, context, i, mode)
X = tf.placeholder(tf.float32, [None, len(x_train.shape)])
Y = tf.placeholder(tf.float32, [None, len(y_train.shape)])
keep_prob = tf.placeholder(tf.float32)

logits = net_MRCNN(X)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.square(Y - logits), reduction_indices=[1])
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss_op)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        for bs_num in range(int(len(x_train) / batch_size) - 1):
            batch_x, batch_y = x_train[bs_num*batch_size: (bs_num+1)*batch_size], y_train[bs_num*batch_size: (bs_num+1)*batch_size]
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: x_test,
                                      Y: y_test,
                                      keep_prob: 1.0}))
