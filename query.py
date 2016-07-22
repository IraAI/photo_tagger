import os
import re
import itertools
import pickle

import pandas as pd
import numpy as np
import cv2

from update_DB import Update_DB

class Query(object):
    DB_PATH = 'action.csv'  # def. relative path to CSV file
    COLUMNS = ['img', 'act_class']  # columns to read from CSV
    VALUES_COLUMN = 'act_class'  # column that contains Values
    PREVIEW_FILE = 'preview.png'


    FEATURES_LAYER = 'fc8'
    SIMILARITY_THRESH = 1.73    # all the images that has disatances 
                                # between them LOWER than this are SIMILAR
    FEATURE_MATRIX_PICKLE = 'featureMatrix2.pickle'
    
    def __init__(self, db_path=DB_PATH, sep=';'):
        self.prj_root = os.getcwd() # project root, current folder
        self.db_path = os.path.join(self.prj_root, self.DB_PATH)
        # load DB
        self.df = pd.read_csv(db_path, usecols=self.COLUMNS, sep=sep)

        # init caffe network and upload dataset
        self.udb = Update_DB()
        self.udb.loadImages()
        self.udb.calculateMean()
        self.udb.preprocess_data()

    def get(self, query_str):
        """
        Get images that satisfies the text query.
        """
        # make regex pattern of query_str
        re_pattern = self.make_regex(query_str)
        # get idxs of rows
        filtered_idxs = self.df[self.VALUES_COLUMN] \
                            .str.contains( re_pattern )
        # get rows
        query_result = self.df[ filtered_idxs ]
        
        file_list = query_result[ self.COLUMNS[0] ].values.flatten()
        labels_list = query_result[ self.COLUMNS[1] ].values.flatten()

        return file_list, labels_list

    def get_similar(self, file_name, max_num=5):
        """
        Get images that are similar to the input one.
        Args:
            img -- 
            max_num -- how many similar to get
            (max_num == -5 gives last 5 - most different ones)
        Returns:
            similar filenames
            different filenames
        """
        # preprocess new img and make forward pass
        img = cv2.imread(os.path.join(self.prj_root, file_name))

        # print(self.prj_root)
        # print(img[:20])

        img_preprocessed = self.udb.preprocess_image(img)
        self.udb.net.blobs['data'].data[...] = img_preprocessed
        self.udb.net.forward()
        # get feature matrix
        new_feature_vec = self.udb.net.blobs[self.FEATURES_LAYER].data
        # print(fm.shape)
        # compare features for cur img and get the list of similar ones
        similar_imgs_idxs = self.find_similar(new_feature_vec, self.SIMILARITY_THRESH)[:max_num]
        
        # print(similar_imgs_idxs)
        # print(self.udb.img_classes)

        file_list = [self.udb.img_filenames[idx] for idx in similar_imgs_idxs]
        labels_list = []#[self.udb.img_classes[idx] for idx in similar_imgs_idxs]

        return file_list, labels_list

    def make_regex(self, query_str, separator=' '):
        """
        Parse query str into regex pattern.
        """
        if isinstance(query_str, str):
            words_lst = self._extract_words(query_str.lower(), separator)
            # create regex ( r'red|shopping|cart' ) pattern
            # to search for specific complete words
            re_pattern = r'\b' + r'\b|\b'.join(words_lst) + r'\b'
        else:
            # match nothing
            re_pattern = r'$^'

        return re_pattern

    def view_result(self, thumbnails, visualize=True,  save=False):
        """
        View the query results in fancy way.
        Args:
            Thumbnail class instance
        """
        if visualize:
            thumbnails.show()
        if save:
            thumbnails.save_to_file(self.PREVIEW_FILE)

    def find_similar(self, new_feature_vec, threshold):
        """
        fm -- feature matrix
        threshold -- mean value thereshold
        """
        # get dataset images feature matrix
        # fm_dataset = self._read_feature_vec()
        fm_dataset = self.udb.get_features(self.FEATURES_LAYER)

        len_dataset = fm_dataset.shape[0]
        sim_lst = np.zeros([len_dataset, 2])

        def _calc_rmse(fm1, fm2):
            return np.sqrt(np.mean((fm1-fm2)**2))

        for i in range(len_dataset):
            sim_lst[i,0] = i
            sim_lst[i,1] = _calc_rmse(fm_dataset[i,:], new_feature_vec[0])
            
        sim_df = pd.DataFrame(sim_lst)
        sim_df[0] = sim_df[0].astype(int)

        sim_df = sim_df.sort_values([1])
        sim_df_selected = sim_df.loc[sim_df[1] <  threshold]

        print(sim_df[1])

        sim_lst = sim_df_selected[0].values

        # print(sim_lst)

        return sim_lst

    def _read_feature_vec(self):
        with open(self.FEATURE_MATRIX_PICKLE, 'rb') as handle:
            fm_dataset = pickle.load(handle)

        return fm_dataset 

    @staticmethod
    def _extract_words(s, separator=' '):
        """
        Split string by white_spaces and strips all the punctuation marks.
        """
        EXCLUDE_CHR = '1234567890!%^&*$()_+=-[]{}|,:;\'\"?'
        result = map( lambda x: x.strip(EXCLUDE_CHR), s.split(separator) )

        return result

class Thumbnails(object):
    IMAGES_FOLDER = 'image_data/dataset1/'  # default images folder
    NUM_CHANNELS = 3  # number of image channels (BGR)
    
    def __init__(self, thumb_size=256, rows=3, cols=5):
        self.prj_root = os.getcwd() # project root, current folder
        self.images_root = os.path.join(self.prj_root, self.IMAGES_FOLDER)
        self.thumb_size = thumb_size
        self.rows = rows
        self.cols = cols
        self.height = thumb_size * rows
        self.width = thumb_size * cols
        self.preview = np.zeros((self.height, self.width, self.NUM_CHANNELS), np.uint8)
        self.preview[:,:,:] = 255
        
    def _add_thumb(self, img, y, x, resize_flag=True):
        if resize_flag:
            img = cv2.resize(img, (self.thumb_size, self.thumb_size))
        # proceed only of we have a place to put the thumb
        if (y + self.thumb_size < self.height) | (x + self.thumb_size < self.width):
            self.preview[y:y + self.thumb_size, x:x + self.thumb_size] = img
        
    def _add_text(self, text, y, x):
        font = cv2.FONT_HERSHEY_SIMPLEX
        y += self.thumb_size - 10
        cv2.putText(self.preview,str(text),(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)
        
    def create_preview_raw_imgs(self, img_list, labels_list=[]):
        # current coordinates to put thumbnail to
        cur_x = 0
        cur_y = 0
        # handle the case if we do not supply labels_list
        if labels_list != []:
            iterate_through = zip(labels_list, img_list)
        else:
            iterate_through = enumerate(img_list)

        for label, img in iterate_through:
                self._add_thumb(img, cur_y, cur_x, resize_flag=False)
                self._add_text(label, cur_y, cur_x)
            
                if cur_x + self.thumb_size < self.width:
                    cur_x += self.thumb_size
                else:
                    cur_x = 0
                    if cur_y + self.thumb_size < self.height:
                        cur_y += self.thumb_size
                    else:
                        break

    def create_preview(self, file_list, labels_list=[]):
        # current coordinates to put thumbnail to
        cur_x = 0
        cur_y = 0
        # handle the case if we do not supply labels_list
        if labels_list != []:
            iterate_through = zip(labels_list, file_list)
        else:
            iterate_through = enumerate(file_list)

        for label, file_name in iterate_through:
            try:
                img = cv2.imread(os.path.join(self.images_root, file_name))
                self._add_thumb(img, cur_y, cur_x)
                self._add_text(label, cur_y, cur_x)
            
                if cur_x + self.thumb_size < self.width:
                    cur_x += self.thumb_size
                else:
                    cur_x = 0
                    if cur_y + self.thumb_size < self.height:
                        cur_y += self.thumb_size
                    else:
                        break
            # temporary solution
            except Exception:
                print('Can\'t read {}.'.format(file_name))

    def show(self):
        cv2.imshow('preview', self.preview)
        cv2.waitKey(0)
    
    def save_to_file(self, file_name):
        cv2.imwrite(file_name, self.preview)


if __name__ == '__main__':
    # query = Query('action (copy).csv')
    query = Query()
    # q_string = 'ShoPping,!!! car'
    COMMANDS = {'text' : 'Describe what are you looking for: ', 
                'file': 'Specify the file name: ', 
                'quit':'Quitting...'}

    file_to_compare = 'test.jpg'

    file_list, labels_list = query.get_similar(file_to_compare)
    # file_list, labels_list = query.get('cat')

    thumbs = Thumbnails(thumb_size=256, \
                        rows=3, \
                        cols=5)
    thumbs.create_preview(file_list, labels_list)
    query.view_result(thumbs, visualize=False, save=True)        

    # while True:
    #     command = raw_input('[ COMMAND ] > ')
    #     if command in COMMANDS.keys():
    #         print(COMMANDS[command])

    #         if command == 'text':
    #             query_str = raw_input('[ '+ command +' ] >')
    #             file_list, labels_list = query.get(query_str)
    #             thumbs = Thumbnails()
    #             thumbs.create_preview(file_list, labels_list)
    #             query.view_result(thumbs, save=True)
    #         elif command == 'file':
    #             file_to_compare = str( raw_input('[ '+ command +' ] >') )
    #             file_list, labels_list = query.get_similar(file_to_compare)
    #             thumbs = Thumbnails()
    #             thumbs.create_preview(file_list, labels_list)
    #             query.view_result(thumbs, save=True)                
    #         elif command == 'quit':
    #             cv2.destroyAllWindows()
    #             break
    #     else:
    #         print('No such command, try again.')
    #         continue