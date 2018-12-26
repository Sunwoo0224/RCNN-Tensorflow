import os
import numpy as np
import tensorflow as tf
import pickle
import time
import xlwt

CUDA_VISIBLE_DEVICES = 0
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.93
config_size = 900

train_batch_size = 15
valid_batch_size = 15
test_batch_size = 10
epoch = 15

current_time = str(int(round(time.time() * 1000)))
subject_name = '083013_w_47'

model_name = '1543521347946'  # use it in retraining
# check_pt_loc = '/model'
# summary_loc = '/sumary'

train = False
retrain = False
test = True

if train or retrain:
    # Restore train data
    with open(subject_name + '_data_train', 'rb') as f:
        data_train = pickle.load(f)
        print('loading data_train is ended')
        f.close()
    with open(subject_name + '_labels_train', 'rb') as f:
        labels_train = pickle.load(f)
        print('loading labels_train is ended')
        f.close()
    with open(subject_name + '_bin_labels_train', 'rb') as f:
        bin_labels_train = pickle.load(f)
        print('loading bin_labels_train is ended')
        f.close()

    # Restore valid data
    with open(subject_name + '_data_valid', 'rb') as f:
        data_valid = pickle.load(f)
        print('loading data_valid is ended')
        f.close()
    with open(subject_name + '_labels_valid', 'rb') as f:
        labels_valid = pickle.load(f)
        print('loading labels_valid is ended')
        f.close()
    with open(subject_name + '_bin_labels_valid', 'rb') as f:
        bin_labels_valid = pickle.load(f)
        print('loading bin_labels_valid is ended')
        f.close()

if test:
    # Restore test data
    with open(subject_name + '_data_test', 'rb') as f:
        data_test = pickle.load(f)
        print('loading data_test is ended')
        f.close()
    with open(subject_name + '_labels_test', 'rb') as f:
        labels_test = pickle.load(f)
        print('loading labels_test is ended')
        f.close()
    with open(subject_name + '_bin_labels_test', 'rb') as f:
        bin_labels_test = pickle.load(f)
        print('loading bin_labels_test is ended')
        f.close()

'''
best_ckpt_saver = BestCheckpointSaver(
    save_dir=best_checkpoint_dir,
    num_to_keep=3,
    maximize=True
)
'''


def train_next_batch(num, data, bin_labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    bin_labels_shuffle = [bin_labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(bin_labels_shuffle)


def test_next_batch(num, data, bin_labels, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    bin_labels_shuffle = [bin_labels[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(bin_labels_shuffle), np.asarray(labels_shuffle)


def get_results(x):
    num_PA0 = 0;
    num_PA1 = 0;
    num_PA2 = 0;
    num_PA3 = 0;
    num_PA4 = 0
    cor_PA0 = 0;
    cor_PA1 = 0;
    cor_PA2 = 0;
    cor_PA3 = 0;
    cor_PA4 = 0

    for i in range(len(x[0])):

        if x[2][i] == 0:
            num_PA0 += 1
            if x[0][i] == x[1][i] and x[1][i] == 0:
                cor_PA0 += 1

        elif x[2][i] == 1:
            num_PA1 += 1
            if x[0][i] == x[1][i] and x[1][i] == 1:
                cor_PA1 += 1

        elif x[2][i] == 2:
            num_PA2 += 1
            if x[0][i] == x[1][i] and x[1][i] == 1:
                cor_PA2 += 1

        elif x[2][i] == 3:
            num_PA3 += 1
            if x[0][i] == x[1][i] and x[1][i] == 1:
                cor_PA3 += 1

        elif x[2][i] == 4:
            num_PA4 += 1
            if x[0][i] == x[1][i] and x[1][i] == 1:
                cor_PA4 += 1

    print('num of BL: ', num_PA0, 'acc of BL: ', float(cor_PA0 / num_PA0))
    print('num of PA1: ', num_PA1, 'acc of PA1: ', float(cor_PA1 / num_PA1))
    print('num of PA2: ', num_PA2, 'acc of PA2: ', float(cor_PA2 / num_PA2))
    print('num of PA3: ', num_PA3, 'acc of PA3: ', float(cor_PA3 / num_PA3))
    print('num of PA4: ', num_PA4, 'acc of PA4: ', float(cor_PA4 / num_PA4))


class RCNN:
    def __init__(self):
        pass

    def RCL(self, X):
        with tf.variable_scope("RCL", reuse=tf.AUTO_REUSE):
            W1 = tf.get_variable("RCL_W1", [1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable("RCL_W2", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())

            conv1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1)
            conv2 = tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.conv2d(conv2, W2, strides=[1, 1, 1, 1], padding='SAME')
            conv4 = tf.nn.conv2d(conv3, W2, strides=[1, 1, 1, 1], padding='SAME')

            return conv4

    def build_model(self):
        print('start build')

        ## Variables for input, output
        X = tf.placeholder(tf.float32, [None, 30, config_size, 3], name='X')
        Y = tf.placeholder(tf.uint8, [None, 1], name='Y')

        nb_class = 2
        Y_one_hot = tf.one_hot(Y, nb_class)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_class])

        ## Convolution Layer

        W1 = tf.get_variable("W1", [3, 3, 3, 256], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([256]))
        # L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME', name='Conv1') + b1)
        L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME', name='Conv1') + b1
        L1 = tf.layers.batch_normalization(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME', name='MaxPool1')

        ## Recurrent Convolution Layer

        # L2 = tf.nn.relu(self.RCL(L1))
        L2 = self.RCL(L1)
        L2 = tf.nn.max_pool(L2, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME', name='MaxPool2')

        # L3 = tf.nn.relu(self.RCL(L2))
        L3 = self.RCL(L2)
        L3 = tf.nn.max_pool(L3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', )

        # L4 = tf.nn.relu(self.RCL(L3))
        L4 = self.RCL(L3)
        L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # L5 = tf.nn.relu(self.RCL(L4))
        L5 = self.RCL(L4)
        L5 = tf.nn.max_pool(L5, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

        ## Fully Connected Layer

        A = int(L5.get_shape()[1])
        B = int(L5.get_shape()[2])
        L5 = tf.reshape(L5, [-1, A * B * 256])
        W_ = tf.get_variable("W_fc", [A * B * 256, 2],
                             initializer=tf.contrib.layers.xavier_initializer())
        b_ = tf.Variable(tf.random_normal([2]))
        # logits = tf.nn.relu(tf.matmul(L5, W_) + b_ , name='model')
        logits = tf.nn.bias_add(tf.matmul(L5, W_), b_, name='model')

        global_step = tf.Variable(0, trainable=False, name='global_step')
        ## Loss, Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='cost')
        optimizer = tf.train.AdamOptimizer(0.001, name='optimizer').minimize(cost, global_step=global_step)

        ## Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ## Summary to use Tensorboard
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('accuracy', accuracy)

        ## train
        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('summary/train-' + current_time, sess.graph)
            valid_writer = tf.summary.FileWriter('summary/valid-' + current_time, sess.graph)
            tf.initialize_all_variables().run()

            batch_size = train_batch_size

            summary_step = 0

            num_data = len(labels_train)
            num_batch = int(num_data / batch_size)

            saver = tf.train.Saver()
            name = 'model/' + current_time

            for i in range(epoch):
                train_total_cost = 0
                valid_total_cost = 0

                for j in range(num_batch):
                    train_batch_xs, train_batch_bin_ys = train_next_batch(batch_size, data_train, bin_labels_train)

                    train_batch_xs = train_batch_xs.reshape(-1, 30, config_size, 3)
                    # train_batch_bin_ys = train_batch_bin_ys.reshape(-1,1)
                    train_summary, _, train_cost_val = sess.run([merged, optimizer, cost],
                                                                feed_dict={X: train_batch_xs, Y: train_batch_bin_ys})

                    if (j % 5 == 0):
                        valid_batch_xs, valid_batch_bin_ys = train_next_batch(batch_size, data_valid, bin_labels_valid)
                        valid_batch_xs = valid_batch_xs.reshape(-1, 30, config_size, 3)
                        valid_summary, valid_cost_val = sess.run([merged, cost],
                                                                 feed_dict={X: valid_batch_xs, Y: valid_batch_bin_ys})

                        train_writer.add_summary(train_summary, summary_step)
                        valid_writer.add_summary(valid_summary, summary_step)

                        summary_step = summary_step + 1

                        # print('global_step:', tf.train.global_step(sess, global_step))
                        print('num_batch:', j + 1, 'train_cost_val: ', train_cost_val, 'valid_cost_val: ',
                              valid_cost_val)

                        valid_total_cost += valid_cost_val

                    train_total_cost += train_cost_val

                train_avg_cost = float(train_total_cost / num_batch)
                valid_avg_cost = float(valid_total_cost / int(num_batch / 5))

                print('#################################################################')
                print('Epoch: ', i + 1, 'Avg train_cost: ', train_avg_cost, 'Avg valid_cost: ', valid_avg_cost)
                print('#################################################################')
                # train_writer.add_summary(summary, i)
                # best_ckpt_saver.handle(train_avg_cost + valid_avg_cost, sess, i)
                saver.save(sess, name + '/my_model', global_step=global_step)

    def retrain_model(self, model_name):

        batch_size = train_batch_size
        num_data = len(labels_train)
        num_batch = int(num_data / batch_size)
        total_correct = 0.0

        checkpoint_file = tf.train.latest_checkpoint('model/' + model_name)
        tf.reset_default_graph()

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())  # Before running session, variables should be initialized

            ## Load a stored RCNN model
            saved_model = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saved_model.restore(sess, checkpoint_file)

            graph = tf.get_default_graph()

            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")

            global_step = graph.get_tensor_by_name('global_step:0')
            cost = graph.get_tensor_by_name('cost:0')
            optimizer = graph.get_tensor_by_name('optimizer:0')
            my_model = graph.get_tensor_by_name('model:0')

            print('global_step:', tf.train.global_step(sess, global_step))
            summary_step = tf.train.global_step(sess, global_step)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('summary/train-' + model_name, sess.graph)
            valid_writer = tf.summary.FileWriter('summary/valid-' + model_name, sess.graph)

            saver = tf.train.Saver()
            name = 'model/' + model_name

            for i in range(epoch):
                total_cost = 0
                for j in range(num_batch):
                    train_batch_xs, train_batch_ys = train_next_batch(batch_size, data_train, labels_train)
                    train_batch_xs = train_batch_xs.reshape(-1, 30, config_size, 3)

                    train_summary, _, train_cost_val = sess.run([merged, optimizer, cost],
                                                                feed_dict={X: train_batch_xs, Y: train_batch_ys})

                    if (j % 5 == 0):
                        valid_batch_xs, valid_batch_ys = train_next_batch(batch_size, data_valid, labels_valid)
                        valid_batch_xs = valid_batch_xs.reshape(-1, 30, config_size, 3)
                        valid_summary, valid_cost_val = sess.run([merged, cost],
                                                                 feed_dict={X: valid_batch_xs, Y: valid_batch_ys})

                        train_writer.add_summary(train_summary, summary_step)
                        valid_writer.add_summary(valid_summary, summary_step)
                        summary_step = summary_step + 1

                        print('num_batch:', j, 'train_cost_val: ', train_cost_val, 'valid_cost_val: ', valid_cost_val)

                    total_cost += train_cost_val

                print('Epoch: ', i + 1, 'Avg cost: ', float(total_cost / num_batch))
                # train_writer.add_summary(summary, i)
                saver.save(sess, name + '/my_model', global_step=global_step)
                # self.train_data, self.train_label = np.random.shuffle(zip(self.train_data, self.train_label))

    def test_model(self, model_name):

        batch_size = test_batch_size
        num_data = len(labels_test)
        num_batch = int(num_data / batch_size)
        total_correct = 0.0

        result1 = []
        result2 = []
        result3 = []

        checkpoint_file = tf.train.latest_checkpoint('model/' + model_name)
        # checkpoint_file = tf.train.load_checkpoint('checkpoint/my_model-49.index')
        # print(checkpoint_file)
        tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())  # Before running session, variables should be initialized

            ## Load a stored RCNN model
            saved_model = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            graph = tf.get_default_graph()
            saved_model.restore(sess, checkpoint_file)

            my_model = graph.get_tensor_by_name('model:0')
            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            # Z = graph.get_tensor_by_name("Z:0")
            # Z = tf.Variable([None,1],trainable=False)

            nb_class = 2
            Y_one_hot = tf.one_hot(Y, nb_class)
            Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_class])

            for i in range(num_batch):
                test_batch_xs, test_batch_bin_ys, test_batch_ys = test_next_batch(batch_size, data_test,
                                                                                  bin_labels_test, labels_test)
                test_batch_xs = test_batch_xs.reshape(-1, 30, config_size, 3)

                print(sess.run(tf.argmax(my_model, 1), feed_dict={X: test_batch_xs}))
                print(sess.run(tf.argmax(Y_one_hot, 1), feed_dict={Y: test_batch_bin_ys}))
                # print(test_batch_ys)

                result1.append(sess.run(tf.argmax(my_model, 1), feed_dict={X: test_batch_xs}))
                result2.append(sess.run(tf.argmax(Y_one_hot, 1), feed_dict={Y: test_batch_bin_ys}))
                result3.append(test_batch_ys)

                is_correct = tf.equal(tf.argmax(my_model, 1), tf.argmax(Y_one_hot, 1))
                acc = tf.reduce_sum(tf.cast(is_correct, tf.float32))

                num_correct = acc.eval(feed_dict={X: test_batch_xs, Y: test_batch_bin_ys})
                # _, cost_val = sess.run([my_optimizer, my_cost], feed_dict={X: batch_xs, Y: batch_ys})

                print('num_batch:', i, 'num_correct: ', num_correct, 'batch_accuracy: ',
                      float(num_correct / batch_size))
                total_correct += num_correct

            print('test_acc: ', total_correct / num_data)

            result1 = (np.array(result1).reshape(1, -1)).tolist()
            result2 = (np.array(result2).reshape(1, -1)).tolist()
            result3 = (np.array(result3).reshape(1, -1)).tolist()

            result = []
            result.append([y for x in result1 for y in x])
            result.append([y for x in result2 for y in x])
            result.append([y for x in result3 for y in x])

            get_results(result)


if __name__ == '__main__':
    rcnn = RCNN()
    if train:
        rcnn.build_model()
    elif retrain:
        rcnn.retrain_model(model_name)
    elif test:
        rcnn.test_model(model_name)
