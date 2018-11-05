import read_data
import os
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

config_size = 8000
class RCNN:
    def __init__(self,X,Y):
        self.data = X
        self.label = Y
        #print(np.shape(self.label))


    def RCL(self,X):
        W1 = tf.Variable(tf.random_normal([1,1,256,256], stddev=0.01))
        W2 = tf.Variable(tf.random_normal([3,3,256,256], stddev=0.01))

        conv1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1],padding='SAME')
        conv1 = tf.layers.batch_normalization(conv1)
        conv2 = tf.nn.conv2d(conv1, W2, strides=[1,1,1,1],padding='SAME')
        conv3 = tf.nn.conv2d(conv2, W2, strides=[1,1,1,1],padding='SAME')
        conv4 = tf.nn.conv2d(conv3, W2, strides=[1,1,1,1],padding='SAME')

        return conv4

    def build_model(self):

        X = tf.placeholder(tf.float32, [None,30,config_size,3]) #45: config_size
        Y = tf.placeholder(tf.float32, [None,5])

        ## Convolution Layer
        W1 = tf.Variable(tf.random_normal([3,3,3,256], stddev=0.01))
        L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
        L1 = tf.layers.batch_normalization(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1,4,1,1], strides=[1,4,1,1], padding='SAME')

        ## Recurrent Convolution Layer
        L2 = self.RCL(L1)
        L2 = tf.nn.max_pool(L2, ksize=[1,4,1,1], strides=[1,4,1,1], padding='SAME')

        L3 = self.RCL(L2)
        L3 = tf.nn.max_pool(L3, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

        L4 = self.RCL(L3)
        L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        L5 = self.RCL(L4)
        L5 = tf.nn.max_pool(L5, ksize=[1,1,1,1], strides=[1,1,1,1], padding='SAME')

        ## Fully Connected Layer
        L5 = tf.reshape(L5, [-1, 1 * 6 * 256])  #config_size 바뀌면 바꿔야함
        W_ = tf.Variable(tf.random_normal([1 * 6 * 256, 5], stddev=0.01))#config_size 바뀌면 바꿔야함


        model = tf.matmul(L5,W_)
        cost = tf.reduce_mean(tf.square(model - Y))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        ## train
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            batch_size = 100
            epoch = 10
            num_data = 13662
            num_batch = int(num_data/batch_size)

            encoder = OneHotEncoder()


            for i in range(epoch):
                total_cost = 0
                for j in range(num_batch):
                    batch_xs = (self.data[j * batch_size: j * batch_size + batch_size]).reshape(-1,30,config_size,3) #config_size에 따라 바꿔야
                    #print(np.shape(batch_xs))
                    batch_ys = np.array(self.label[j * batch_size: j * batch_size + batch_size])
                    #print(np.shape(batch_ys))
                    #batch_ys = tf.one_hot(batch_ys,5)
                    batch_ys = (encoder.fit_transform(batch_ys)).toarray()
                    #print(np.shape(batch_ys))
                    #batch_ys = (batch_ys).reshape((100,-1))

                    _,cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
                    print('num_batch:', j ,'cost_val: ',cost_val)
                    total_cost += cost_val
                print('Epoch: ', i + 1, 'Avg cost: ', float(total_cost/num_batch))


if __name__ == "__main__":
    current_folder = os.getcwd()
    print('Location of Data: ', current_folder)

    # X, Y = read_data.getData(current_folder + '/112909_w_20', config_size)
    with open('X', 'rb') as f1:
        X = pickle.load(f1)
    with open('Y', 'rb') as f2:
        Y = pickle.load(f2)
    print('Data is ready')

    #print(np.shape(X))
    #print(np.shape(Y))
    #print(len(X))
    #print(len(Y))
    X = np.array(X)
    Y = np.array(Y)

    #Y = tf.one_hot(Y, 5)
    X = X.reshape(-1,30*config_size*3) #data함수에서 data갯수 return 받으면 좋을듯
    Y = Y.reshape(-1,1)
    #print(np.shape(Y))
    #print(np.shape(X))
    #Y = Y.reshape(-1,5)
    #X = tf.reshape(X,[-1,30*45*3])
    #Y = tf.reshape(Y,[-1,5])

    data_train, data_test, labels_train, labels_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    rcnn = RCNN(data_train, labels_train)
    rcnn.build_model()

