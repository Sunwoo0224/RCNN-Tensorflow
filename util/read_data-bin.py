import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

config_size = 900
resize_size = 10

subject_name = '081714_m_36'
test_ratio = 0.2
valid_ratio = 0.25


## Function for calculating average of width and height
def calculate(foldername):
    width = []
    height = []

    for filename in os.listdir(foldername):  # one subject has 100 videos
        if filename != '.DS_Store':
            subfolder = os.path.join(foldername, filename)
            for name in os.listdir(subfolder):  # one video is composed of 138 frame images
                if name == 'fit':
                    name = os.path.join(subfolder, name)
                    for val in os.listdir(name):  # Deal with image files
                        if val[-3:] == 'jpg':
                            filename = os.path.join(name, val)
                            img = Image.open(filename)
                            width.append(img.size[0])
                            height.append(img.size[1])

    width = np.array(width)
    height = np.array(height)

    return int(np.mean(width)), int(np.mean(height)), len(width)

## Function for image flattening
def flattening(img, config_size):
    img = np.array(img)[..., :3]
    flatten_img = []

    for i in range(3):
        flatten_img.append(img[:, :, i].reshape(-1)[:config_size])

    return flatten_img

## Fuction changing float to int
def floatToInt(list):
    for index,val in enumerate(list):
        list[index] = int(val)
    return list

## Numerify the pain level
def labelToBin(string):
    if string == 'BL1':
        return 0
    elif string == 'PA1':
        return 1
    elif string == 'PA2':
        return 1
    elif string == 'PA3':
        return 1
    elif string == 'PA4':
        return 1

def labelToNum(string):
    if string == 'BL1':
        return 0
    elif string == 'PA1':
        return 1
    elif string == 'PA2':
        return 2
    elif string == 'PA3':
        return 3
    elif string == 'PA4':
        return 4


## Get Data from a folder
def getData(foldername, resize_size, config_size):
    Z = []
    Y = []                 # answer
    X = []                 # input
    video_num =0

    # calculate average of width, height
    width, height, total_num = calculate(foldername)
    print('Avg width:', width, ' Avg height:', height, ' Total num of data:', total_num)

    for filename in os.listdir(foldername):         # one subject has 100 videos
        if filename != '.DS_Store':
            pain_level = filename.split("-")[0]
            subfolder = os.path.join(foldername, filename)

            for name in os.listdir(subfolder):      # one video is composed of 138 frame images
                if name == 'fit':
                    video_num = video_num + 1
                    frame_num = 0

                    x = []                              # list for storing sequence of H(=30) number of element of _x
                    _x = []                             # list for storing extracted image data from a video by frame
                    name = os.path.join(subfolder, name)

                    for val in os.listdir(name):        # Deal with image files
                        if val[-3:] == 'jpg':
                            frame_num = frame_num + 1
                            filename = os.path.join(name, val)
                            img = Image.open(filename)

                            # resize
                            resize_img = img.resize((int(width/resize_size), int(height/resize_size)), Image.ANTIALIAS)

                            # flatten
                            flatten_img = flattening(resize_img, config_size)

                            # transpose
                            _x.append(np.transpose(flatten_img))

                    print('video_num:',video_num,'frame_num',frame_num,'name',name)

                    for i in range(len(_x)):
                        tmp = []
                        if i < 29:
                            for j in range(29-i):
                                tmp.append(np.zeros((np.shape(flatten_img)[1],3), dtype=int))
                            for j in range(i+1):
                                tmp.append(np.array(_x[:][j]))
                        else:
                            for j in range(i-29,i+1):
                                tmp.append(np.array(_x[:][j]))
                        X.append(np.uint8(tmp))
                        Y.append(labelToBin(pain_level))
                        Z.append(labelToNum(pain_level))
                    #print('Processing is ended for video ',video_num)
    return X, Y, Z

if __name__ == '__main__':

    X, Y, Z= getData('/Users/sunwoo/Desktop/'+subject_name,resize_size,config_size)
    print('Data is ready')

    X = np.array(X)
    X = X.reshape(-1, 30 * config_size * 3)
    Y = np.array(Y)
    Y = Y.reshape(-1, 1)
    Z = np.array(Z)
    Z= Z.reshape(-1, 1)

    print('Cross Validation')
    train_data_, test_data, train_bin_labels_, test_bin_labels, train_labels_, test_labels = train_test_split(X, Y, Z, test_size=test_ratio, random_state=42)
    train_data, valid_data, train_bin_labels, valid_bin_labels, train_labels, valid_labels = train_test_split(train_data_, train_bin_labels_, train_labels_, test_size=valid_ratio, random_state=42)

    # Save train data
    with open(subject_name + '_data_train', 'wb') as f:
        pickle.dump(train_data, f)
        print('saving data_train is ended')
        f.close()
    with open(subject_name + '_labels_train', 'wb') as f:
        pickle.dump(train_labels, f)
        print('saving labels_train is ended')
        f.close()
    with open(subject_name + '_bin_labels_train', 'wb') as f:
        pickle.dump(train_bin_labels, f)
        print('saving bin_labels_train is ended')
        f.close()

    # Save valid data
    with open(subject_name + '_data_valid', 'wb') as f:
        pickle.dump(valid_data, f)
        print('saving data_valid is ended')
        f.close()
    with open(subject_name + '_labels_valid', 'wb') as f:
        pickle.dump(valid_labels, f)
        print('saving labels_valid is ended')
        f.close()
    with open(subject_name + '_bin_labels_valid', 'wb') as f:
        pickle.dump(valid_bin_labels, f)
        print('saving labels_valid is ended')
        f.close()

    # Save test data
    with open(subject_name + '_data_test', 'wb') as f:
        pickle.dump(test_data, f)
        print('saving data_test is ended')
        f.close()
    with open(subject_name + '_labels_test', 'wb') as f:
        pickle.dump(test_labels, f)
        print('saving labels_test is ended')
        f.close()
    with open(subject_name + '_bin_labels_test', 'wb') as f:
        pickle.dump(test_bin_labels, f)
        print('saving bin_labels_test is ended')
        f.close()

