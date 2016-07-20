import os

import pandas as pd
# import cv2

class Query(object):
    DB_PATH = 'actions.csv'  # relative path to CSV file
    COLUMNS = ['img', 'act_class']  # columns to read from CSV
    VALUES_COLUMN = 'act_class'  # column that contains Values
    
    def __init__(self, db_path=DB_PATH):
        self.df = pd.read_csv(db_path, usecols=self.COLUMNS)
        self.file_names = []
        
    @staticmethod
    def _parse(query_str, separator=' '):
        """Parse query str into list."""
        if isinstance(query_str, str):
            query_lst = query_str.split(separator)
        return query_lst
        
    def get(self, query_str):
        """Process the query.
        Args:
            query_str -- string from client's UI
        Returns:
            paths -- list of paths
        """
        query_lst = self._parse(query_str)
        print(query_lst)
        self.file_names = self.df[ self.df[self.VALUES_COLUMN].isin(query_lst) ]
        
    def view_result(self):
        """View the query results in fancy way."""
        print(self.file_names)