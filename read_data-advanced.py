import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

config_size = 900
resize_size = 10

subject_name = '112909_w_20'
test_ratio = 0.3
valid_ratio = 0.2

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
    train_X = [] ; valid_X = [] ; test_X = []
    train_bin_Y = [] ; valid_bin_Y = [] ; test_bin_Y =[]
    train_Y = [] ; valid_Y = [] ; test_Y = []

    video_num =0

    # calculate average of width, height
    width, height, total_num = calculate(foldername)
    print('Avg width:', width, ' Avg height:', height, ' Total num of data:', total_num)

    filelist = np.asarray(os.listdir(foldername))
    np.random.shuffle(filelist)
    num_for_cv = 0

    for filename in filelist:         # one subject has 100 videos
        if filename != '.DS_Store':
            num_for_cv += 1

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

                        ## Cross-Validation
                        # Make test set
                        if num_for_cv <= 100 * test_ratio:
                            test_X.append(np.uint8(tmp))
                            test_bin_Y.append(np.uint8(labelToBin(pain_level)))
                            test_Y.append(np.uint8(labelToNum(pain_level)))

                        else:
                            train_X.append(np.uint8(tmp))
                            train_bin_Y.append(np.uint8(labelToBin(pain_level)))
                            train_Y.append(np.uint8(labelToNum(pain_level)))
                        '''    
                        # Make Valid set
                        elif num_for_cv <= 100 * test_ratio + 100 * valid_ratio:
                            valid_X.append(np.uint8(tmp))
                            valid_bin_Y.append(np.uint8(labelToBin(pain_level)))
                            valid_Y.append(np.uint8(labelToNum(pain_level)))
                        
                        # Make train set
                        else:
                            train_X.append(np.uint8(tmp))
                            train_bin_Y.append(np.uint8(labelToBin(pain_level)))
                            train_Y.append(np.uint8(labelToNum(pain_level)))
                        '''

    train_X = np.array(train_X); train_bin_Y = np.array(train_bin_Y); train_Y = np.array(train_Y)
    valid_X = np.array(valid_X); valid_bin_Y = np.array(valid_bin_Y); valid_Y = np.array(valid_Y)
    test_X = np.array(test_X); test_bin_Y = np.array(test_bin_Y); test_Y = np.array(test_Y)

    train_X = train_X.reshape(-1, 30 * config_size * 3)
    valid_X = valid_X.reshape(-1, 30 * config_size * 3)
    test_X = test_X.reshape(-1, 30 * config_size * 3)

    train_bin_Y = train_bin_Y.reshape(-1, 1)
    valid_bin_Y = valid_bin_Y.reshape(-1, 1)
    test_bin_Y = test_bin_Y.reshape(-1, 1)

    train_Y = train_Y.reshape(-1, 1)
    valid_Y = valid_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)

    # Save train data
    with open(subject_name + '_data_train', 'wb') as f:
        pickle.dump(train_X, f)
        print('saving data_train is ended')
        f.close()
    with open(subject_name + '_labels_train', 'wb') as f:
        pickle.dump(train_Y, f)
        print('saving labels_train is ended')
        f.close()
    with open(subject_name + '_bin_labels_train', 'wb') as f:
        pickle.dump(train_bin_Y, f)
        print('saving bin_labels_train is ended')
        f.close()
    print()
    print('#########################')
    """
    # Save valid data
    with open(subject_name + '_data_valid', 'wb') as f:
        pickle.dump(valid_X, f)
        print('saving data_valid is ended')
        f.close()
    with open(subject_name + '_labels_valid', 'wb') as f:
        pickle.dump(valid_Y, f)
        print('saving labels_valid is ended')
        f.close()
    with open(subject_name + '_bin_labels_valid', 'wb') as f:
        pickle.dump(valid_bin_Y, f)
        print('saving labels_valid is ended')
        f.close()
    print()
    """
    print('#########################')
    # Save test data
    with open(subject_name + '_data_test', 'wb') as f:
        pickle.dump(test_X, f)
        print('saving data_test is ended')
        f.close()
    with open(subject_name + '_labels_test', 'wb') as f:
        pickle.dump(test_Y, f)
        print('saving labels_test is ended')
        f.close()
    with open(subject_name + '_bin_labels_test', 'wb') as f:
        pickle.dump(test_bin_Y, f)
        print('saving bin_labels_test is ended')
        f.close()
    print()
    print('#########################')

if __name__ == '__main__':

    getData('/Users/sunwoo/Desktop/'+subject_name,resize_size,config_size)
    print('Data is ready')

