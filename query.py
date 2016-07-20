import os
import re

import pandas as pd
# import cv2

class Query(object):
    DB_PATH = 'action.csv'  # def. relative path to CSV file
    COLUMNS = ['img', 'act_class']  # columns to read from CSV
    VALUES_COLUMN = 'act_class'  # column that contains Values
    
    def __init__(self, db_path=DB_PATH, sep=';'):
        self.df = pd.read_csv(db_path, usecols=self.COLUMNS, sep=sep)
                
    def get(self, query_str):
        """Process the query."""
        re_pattern = self.make_regex(query_str)

        filtered_idxs = self.df[self.VALUES_COLUMN] \
                            .str.contains( re_pattern )
            
        self.query_result = self.df[ filtered_idxs ]
        
        return self.query_result

    def make_regex(self, query_str, separator=' '):
        """Parse query str into regex pattern."""
        if isinstance(query_str, str):
            words_lst = self._extract_words(query_str.lower(), separator)
            
            # create regex ( r'red|shopping|cart' ) pattern to make a query on Vals column
            re_pattern = r'\b' + r'\b|\b'.join(words_lst) + r'\b'
            
        else:
            # match nothing
            re_pattern = r'$^'
        return re_pattern

    def view_result(self):
        """View the query results in fancy way."""
        print(self.query_result)
        self.query_result.head()

    @staticmethod
    def _extract_words(s, separator=' '):
        """Split string by white_spaces and strips all the punctuation marks."""
        EXCLUDE_CHR = '1234567890!%^&*$()_+=-[]{}|,:;\'\"?'
        result = map( lambda x: x.strip(EXCLUDE_CHR), s.split(separator) )

        return result