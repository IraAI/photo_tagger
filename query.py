import os
import re

import pandas as pd
import numpy as np
import cv2


class Query(object):
    DB_PATH = 'action.csv'  # def. relative path to CSV file
    COLUMNS = ['img', 'act_class']  # columns to read from CSV
    VALUES_COLUMN = 'act_class'  # column that contains Values
    PREVIEW_FILE = 'preview.png'
    
    def __init__(self, db_path=DB_PATH, sep=';'):
        self.prj_root = os.getcwd() # project root, current folder
        self.db_path = os.path.join(self.prj_root, self.DB_PATH)
        # load DB
        self.df = pd.read_csv(db_path, usecols=self.COLUMNS, sep=sep)
        # create thumbnails sheet
        self.thumbs = Thumbnails()
                
    def get(self, query_str):
        """
        Process the query.
        """
        # make regex pattern of query_str
        re_pattern = self.make_regex(query_str)
        # get idxs of rows
        filtered_idxs = self.df[self.VALUES_COLUMN] \
                            .str.contains( re_pattern )
        # get rows
        self.query_result = self.df[ filtered_idxs ]
        
        return self.query_result

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

    def view_result(self, visualize=True, save=False):
        """
        View the query results in fancy way.
        """
        file_list = self.query_result[ self.COLUMNS[0] ].values.flatten()
        labels_list = self.query_result[ self.COLUMNS[1] ].values.flatten()

        self.thumbs.create_preview(file_list, labels_list)
        if visualize:
            self.thumbs.show()
        if save:
            self.thumbs.save_to_file(self.PREVIEW_FILE)

    @staticmethod
    def _extract_words(s, separator=' '):
        """
        Split string by white_spaces and strips all the punctuation marks.
        """
        EXCLUDE_CHR = '1234567890!%^&*$()_+=-[]{}|,:;\'\"?'
        result = map( lambda x: x.strip(EXCLUDE_CHR), s.split(separator) )

        return result

class Thumbnails(object):
    IMAGES_FOLDER = 'image_data/new_images/'  # default images folder
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
        
    def _add_thumb(self, img, y, x):

            img = cv2.resize(img, (self.thumb_size, self.thumb_size))
            # proceed only of we have a place to put the thumb
            if (y + self.thumb_size < self.height) | (x + self.thumb_size < self.width):
                self.preview[y:y + self.thumb_size, x:x + self.thumb_size] = img
        
    def _add_text(self, text, y, x):
        font = cv2.FONT_HERSHEY_SIMPLEX
        y += self.thumb_size - 10
        cv2.putText(self.preview,str(text),(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)
        
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
        while True:
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break
    
    def save_to_file(self, file_name):
        cv2.imwrite(file_name, self.preview)


if __name__ == '__main__':
    # query = Query('action (copy).csv')
    query = Query()
    q_string = 'ShoPping,!!! car'
    # q_string = raw_input('[keywords]> ')
    query.get(q_string)
    query.view_result()