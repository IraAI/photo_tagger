import caffe
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class Update_DB(object):
    DB_PATH = 'categories.csv'  # relative path to CSV file
    #DB_PATH = 'categories2.csv'  # relative path to CSV file
    COLUMNS = ['img', 'act_class']  # columns to read from CSV
    VALUES_COLUMN = 'act_class'  # column that contains Values
    
    MODEL_CONFIG_PATH = 'model/bvlc_reference_caffenet/deploy.prototxt'
    # binary pretrained model
    MODEL_WEIGTHS_PATH = 'model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # labels from ImageNet, 1000 categories
    LABELS_PATH = 'image_data/labels1/synset_words.txt'
    # dataset
    IMAGES_FOLDER_PATH = 'image_data/dataset1/'
    #IMAGES_FOLDER_PATH = 'image_data/dataset2/'
    # CageNet takes as input images227x227
    WIDTH = 227
    HEIGHT = 227
    # mean for normalizing dataset
    # MEAN_VALS = np.array([104.00698793, 116.66876762, 122.67891434])
    LABELS_CNT = 5
    CHUNK_SIZE = 100
    FEATURE_LAYER = 'fc8'
    
    def __init__(self):
        #caffe.set_device(0)  # if we have multiple GPUs, pick the first one
        #caffe.set_mode_gpu()        
        
        self.prj_root = os.getcwd() # project root, current folder
        self.model_config = os.path.join(self.prj_root, self.MODEL_CONFIG_PATH)
        self.model_weights = os.path.join(self.prj_root, self.MODEL_WEIGTHS_PATH)
        # load network
        self.net = caffe.Net(self.model_config, self.model_weights, caffe.TEST)
        self.net.blobs['data'].reshape(1, 3, self.WIDTH, self.HEIGHT)
        self.labels_file = os.path.join(self.prj_root, 'image_data/labels1/synset_words.txt')
        # get class names from labels file
        self.class_names = np.loadtxt(self.labels_file, str, delimiter='\t')
        self.images = []
        self.img_filenames = []
        self.img_classes = []
        self.avg = np.array([0,0,0])
        #self.img_folder =  os.path.join(self.prj_root, self.IMAGES_FOLDER_PATH)
        #self.img_filenames = os.listdir(img_folder)
        #self.n =len(img_filenames)
        
    #def run(self):
    #  for i in xrange(0, n)
              
    def loadImages(self):
        img_folder =  os.path.join(self.prj_root, self.IMAGES_FOLDER_PATH)
        img_filenames = os.listdir(img_folder)
        n =len(img_filenames)
        # print(img_folder)
        for img_filename in os.listdir(img_folder):
            self.img_filenames.append(img_filename)
            img_fullname = os.path.join(img_folder, img_filename)
            img = cv.imread(img_fullname)
            # cv.imshow('image',img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # print img_filename
            self.images.append(img)
                     
    def calculateMean(self):
        (height, width, channels) = self.images[0].shape
        
        for i in xrange(0, len(self.images)):
            for j in xrange(0, channels):
                self.avg[j] = self.avg[j] + np.average(self.images[i][:,:,j])/len(self.images)
        #print avg
        #print np.average(self.images[0][:,:,0])
        #print np.average(self.images[0][:,:,1])
        #print np.average(self.images[0][:,:,2])
        
    def preprocess_data(self):
        # Data needs to be preprocessed in order to pass it to the network
        # print(img_folder)
        for i in xrange(0,len(self.images)):
            img = self.images[i]
            img = cv.resize(img, (227,227))
            img = img.astype(float)
            img = img - self.avg
            img = img.swapaxes(0, 2).swapaxes(1,2)
            self.images[i]=img
            
        for i in xrange(0,len(self.class_names)):
            blank_index = self.class_names[i].find(" ")
            self.class_names[i] = self.class_names[i][blank_index+1:]
            # print blank_index
    
    def preprocess_image(self, img):
        # Data needs to be preprocessed in order to pass it to the network

        print(type(img))

        img = cv.resize(img, (227,227))
        img = img.astype(float)
        img = img - self.avg
        img = img.swapaxes(0, 2).swapaxes(1,2)

        return img    
        
    def classify(self):
        # pass images to the network, predict tags
        #for img in self.images:
        for i in xrange(0, len(self.images)):
            img = self.images[i]
            self.net.blobs['data'].data[...] = img
            res = self.net.forward()['prob'][0]
            # print self.class_names[np.argmax(res)]
            self.img_classes.append("");
            for j in xrange(0, self.LABELS_CNT):
                self.img_classes[i] = self.img_classes[i] + self.class_names[np.argmax(res)]
                res[np.argmax(res)]= 0
                if j < self.LABELS_CNT-1:
                    self.img_classes[i] = self.img_classes[i] + ", "
            #self.img_classes.append(self.class_names[np.argmax(res)])
            
    def writeCsv(self):
        d = {'img' : self.img_filenames, 'act_class' : self.img_classes}
        df = pd.DataFrame(data=d, columns=['img', 'act_class'])
        df.to_csv(self.DB_PATH, sep=';', header=True, index=False)
        # relative path to CSV file, sep=';', header=True, index=False)
        # print df

    def get_features(self, layer_name):
        # get feature map vector
        num_images = len(self.images)
        num_features = self.net.blobs[layer_name].channels

        feature_matrix = np.zeros((num_images, num_features), np.float64)
        for idx, img in enumerate(self.images):
            self.net.blobs['data'].data[...] = img
            self.net.forward()
            feature_vec = self.net.blobs[layer_name]
            feature_matrix[idx,:] = feature_vec.data

        return feature_matrix
        
        
    def run_feature(self):
        udb = Update_DB()
        udb.loadImages()
        udb.calculateMean()
        udb.preprocess_data()
        fm = udb.get_features(self.FEATURE_LAYER)
        
        print(fm.shape)
        return fm
        