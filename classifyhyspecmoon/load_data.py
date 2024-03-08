'''
Here we will load the data required for our neural net. Processing is available in a seperate package, native to Julia (https://github.com/z-vig/JENVI.jl.git).
'''
import keras
import numpy as np
import h5py as h5

class NeuralNetData():
    def __init__(self,h5path) -> None:
        with h5.File(h5path) as f:
            self.raw_spec = np.zeros(f['VectorDataSets/RawSpectra'].shape)

    def test(self):
        print(self.raw_spec.shape)

