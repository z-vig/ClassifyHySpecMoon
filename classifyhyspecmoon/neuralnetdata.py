'''
Here we will load the data required for our neural net. Processing is available in a seperate package, native to Julia (https://github.com/z-vig/JENVI.jl.git).
'''
import keras
import numpy as np
import h5py as h5
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .create_labels import spa_label

def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5._hl.group.Group,h5._hl.files.File]:
        for key in obj.keys():
            print(f'{sep}-{key}:{obj[key]}')
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(f'{sep}\t-{key}:{obj.attrs[key]}')

def include_data(h5obj,h5obj_path,flip=True):
    arr = np.zeros(h5obj[h5obj_path].shape)
    arr[:] = h5obj[h5obj_path]
    if flip==True:
        arr = arr[:,::-1]
    elif flip=='y':
        arr = arr[::-1,:]
    return arr

class NeuralNetData():
    """
    Class that defines data for input into the neural net

    Methods:
        h5dump() // Allows easy visualization of input h5 data file

    Attributes:
        h5path // Gives the same string that was input to find the h5 file
        rawspec // Gives the raw spectral data
        smoothspec // Gives the smoothed spectral data
    """
    def __init__(self,h5path) -> None:
        self.h5path = h5path
        with h5.File(h5path) as f:

            self.rawspec = include_data(f,'VectorDatasets/RawSpectra',flip=True)
            self.smoothspec = include_data(f,'VectorDatasets/SmoothSpectra_GNDTRU',flip=True)
            self.X = np.moveaxis(self.smoothspec.reshape(self.smoothspec.shape[0],self.smoothspec.shape[1]*self.smoothspec.shape[2]),0,1)
            self.shadowmap = include_data(f,'ShadowMaps/lowsignal_shadows',flip='y')

            self.wvl = f.attrs['smooth_wavelengths']

            self.num_pixels = self.rawspec.shape[1]*self.rawspec.shape[2]

    def h5dump(self) -> None:
        """Displays the entire file tree for the input h5 data file"""
        with h5.File(self.h5path) as f:
            descend_obj(f)
    
    def label_data(self,label_type:'str',refspec_dict:'dict') -> None:
        """
        Method for labeling our data. The possible label types are:
            "Spectral_Angle" // Cosine similarity metric
        """
        self.labeled_data = np.zeros(self.smoothspec.shape[1:])
        print(self.labeled_data.shape)
        label_dict = {'shadow':0}
        label_maps = []
        if label_type == 'Spectral_Angle':
            n = 1
            for key in refspec_dict.keys():
                label_dict[key] = n
                lmap = spa_label(self.smoothspec,refspec_dict[key][0],refspec_dict[key][1],self.shadowmap)
                self.labeled_data[lmap==1] = n
                label_maps.append(lmap)
                n+=1
        elif label_type == None:
            pass

        background_label = np.zeros(self.labeled_data.shape)
        for lmap in label_maps:
            background_label[lmap==1] = 1

        print(n)
        self.labeled_data[background_label==0] = n
        label_dict['background'] = n
        self.labeled_data[self.shadowmap==1] = 0
        self.Y = self.labeled_data.flatten()
        
        self.num_labels = len(np.unique(self.labeled_data))

    def split_train_test(self) -> None:
        """
        Method for splitting our data into testing and training sets
        """
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y,test_size=0.25,random_state=42)

    def minmax_normalization(self,minmaxrange:'tuple') -> None:
        """
        Method for normalizing the data to the minmaxrange
        """
        scaler = MinMaxScaler(minmaxrange,copy=False)
        scaler.fit(self.X_train)
        
        scaler.transform(self.X_train)
        scaler.transform(self.X_test)
        
    

    

        

        

