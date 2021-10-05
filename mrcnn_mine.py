from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.constraints import max_norm
import numpy as np
from sklearn.model_selection import train_test_split
import configs as configs
import pandas as pd


def run_experiment(organism_name, context, i, root):
    X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '__' + organism_name + '.npy', allow_pickle=True)
    Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '__' + organism_name + '.npy', allow_pickle= True)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size),Y] = 1
    Y = b
    X = X.reshape(list(X.shape) + [1])
    X = np.swapaxes(X,1,2)

    PROFILE_COLS = X.shape[1]
    PROFILE_ROWS = X.shape[2]

    test_percent = 0.2
    test_val_percent = 0.5
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)

    W_maxnorm = 3

    model = Sequential()


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
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print (h_fc1_drop)

    W_fc2  = weight_variable([80, 1])
    b_fc2  = bias_variable([1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print (y_conv)
    loss = tf.reduce_mean(tf.square(y - y_conv),reduction_indices=[1])
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)





    model.add(Conv2D(16, kernel_size=(1, 4), input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='VALID', use_bias=True))
    model.add(Reshape((20, 20, 16), input_shape=(PROFILE_ROWS, 1, 16))) #end of first

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  padding='VALID', use_bias=True))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3,3)))

    ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID'

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    print (h_pool2)

    model.add(Conv2D(512, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1,3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    myoptimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=myoptimizer,metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=0, validation_split=0.1)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    step_res = [organism_name, context, 'cpgenie', str(i), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    print(step_res)
    return step_res


root = '/home/csgrads/ssere004/output_cpgenie/'
contexts = ['CG', 'CHG', 'CHH']
res=[]



#residues = [('CHH', 1), ('CHG', 4)]
#for context, i in residues:
#    res.append(run_experiment(organism_name, context, i, root))
#    np.savetxt("GFG_cpgenie.csv", res, delimiter=", ", fmt='% s')

cnfgs = [configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
for cnfg in cnfgs:
    organism_name = cnfg['organism_name']
    for i in range(1,2):
        for context in contexts:
            res.append(run_experiment(organism_name, context, i, root))
            np.savetxt("GFG_cpgenie.csv", res, delimiter=", ", fmt='% s')



def weight_variable(shape):
 initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
 return tf.compat.v1.Variable(initial)


def bias_variable(shape):
 initial = tf.compat.v1.constant(0.1, shape=shape)
 return tf.compat.v1.Variable(initial)
