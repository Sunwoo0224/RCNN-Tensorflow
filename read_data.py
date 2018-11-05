import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

# Function for image resizing
def resize(img, width, height):
    new_width = int(width / 4)
    new_height = int(height / 4)

    resize_img = img.resize((new_width, new_height))
    return resize_img
# Function for calculating average of width and height
def calculate(foldername):
    width = []
    height = []

    for filename in os.listdir(foldername):  # one subject has 100 videos
        if filename != '.DS_Store' and filename != 'BL1-088' and filename[len(filename) - 4:len(
                filename)] != '.mp4' and filename[0:6] != 'ffmpeg':
            subfolder = os.path.join(foldername, filename)

            for name in os.listdir(subfolder):  # one video is composed of 138 frame images
                if name == 'fit':
                    name = os.path.join(subfolder, name)
                    for val in os.listdir(name):  # Deal with image files
                        if val != '.DS_Store':
                            filename = os.path.join(name, val)
                            img = Image.open(filename)

                            width.append(img.size[0])
                            height.append(img.size[1])

    width = np.array(width)
    height = np.array(height)

    return int(np.mean(width)), int(np.mean(height))

# Function for image flattening
def flattening(img):
    img = np.array(img)[..., :3]
    flatten_img = []
    #print(np.shape(img))
    for i in range(3):
        #print(np.shape(img[:,:,i].reshape(-1)[:8000]))
        flatten_img.append(img[:,:,i].reshape(-1)[:8000])
    #flatten_img = np.array(flatten_img[:8000])
    #print(np.shape(flatten_img))
    return flatten_img

# Numerify the pain level
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

# Get Data from a folder
def getData(foldername):
    total_num = 138 * 99   # number of total input data
    Y = []                 # answer
    X = []                 # input
    video_num =0

    # calculate average of width, height
    width, height = calculate(foldername)

    for filename in os.listdir(foldername):         # one subject has 100 videos
        #print(filename)
        if filename != '.DS_Store' and filename != 'BL1-088' and filename[len(filename)-4:len(filename)] != '.mp4' and filename[0:6] != 'ffmpeg':
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
                        if val != '.DS_Store':
                            frame_num = frame_num + 1
                            filename = os.path.join(name, val)
                            img = Image.open(filename)

                            # resize
                            resize_img = resize(img,width,height)
                            # flatten
                            flatten_img = flattening(resize_img)
                            #print(subfolder, val, resize_img.size[0], resize_img.size[1])
                            #print(np.shape(flatten_img)[1])
                            #_x.append(flatten_img)
                            _x.append(np.transpose(flatten_img))
                            '''
                            ## test code
                            for i in range(len(_x)):
                                if np.shape(_x[i])[0] != 45:
                                    print(np.shape(_x[i]))
                            '''
                    print('video_num:',video_num,'frame_num',frame_num,'name',name)

                    for i in range(len(_x)):
                        tmp = []
                        if i < 29:
                            for j in range(29-i):
                                #tmp.append(np.zeros((3,290)))
                                tmp.append(np.zeros((np.shape(flatten_img)[1],3)))
                            for j in range(i+1):
                                #tmp.append(np.array(_x[j]))
                                tmp.append(np.array(_x[:][j]))
                        else:
                            for j in range(i-29,i+1):
                                #tmp.append(np.array(_x[j]))
                                tmp.append(np.array(_x[:][j]))

                        X.append(tmp)
                        Y.append(labelToNum(pain_level))


    #print(video_num)
    return X, Y


if __name__ == "__main__":
    '''
    width, height = calculate('/Users/sunwoo/PycharmProjects/untitled1/112909_w_20')
    print(width , height)
    img = Image.open('/Users/sunwoo/PycharmProjects/untitled1/112909_w_20/BL1-081/fit/crop-1.jpg')
    new_img = resize(img,width,height)
    new_img.show()
    '''

    X, Y = getData('/Users/sunwoo/PycharmProjects/untitled1/112909_w_20')

    with open('X', 'wb') as f1:
        pickle.dump(X,f1)
        f1.close()

    with open('Y','wb') as f2:
        pickle.dump(Y,f2)
        f2.close()
