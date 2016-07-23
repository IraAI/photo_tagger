import numpy as np
import pandas as pd

from update_DB import Update_DB



class Single_Image():
    """ By passing a single image you can receive all the images similar to this
    one.
    """
    
    
    def __init__(self):
        self.udb = Update_DB()
        self.udb.loadImages()
        self.udb.calculateMean()
        self.udb.preprocess_data()

        self.fm = self.udb.get_features('fc8')
        print(self.fm.shape)
        
        
        
    def pass_image(self, single_image):
        num_img = self.fm.shape[0]
        
        
        # create dataframe with index, filename and euclidean distance
        
        self.udb.img_filenames
        print(len(self.udb.img_filenames))
        
        self.df_rmse = pd.DataFrame({'index': range(len(self.udb.img_filenames)), 'filenames':self.udb.img_filenames, 'rmse': np.zeros(len(self.udb.img_filenames))})
        print(self.df_rmse)
        
        single_rmse = self.df_rmse.loc(self.df_rmse['filenames'] == single_image)
        fm_single = self.fm[single_rmse[0]['index']]
        
        
        def _calc_rmse(fm1, fm2):
            return np.sqrt(np.mean((fm1-fm2)**2))
        
        
        for i in range(num_img):
            
            self.df_rmse.ix[i,'rmse'] = _calc_rmse(self.fm[i,:],fm_single)
            
        print(self.df_rmse)
        
    def select_similar(self, threshold):
        
        self.df_rmse_similar = self.df_rmse.loc[self.df_rmse['rmse'] <  threshold]
        print(self.df_rmse_similar)
            
if __name__ == '__main__':
    
    single_image = Single_Image()
    image_name = '2015-03-07 18.50.15.jpg'
    single_image.pass_image(image_name)
    
