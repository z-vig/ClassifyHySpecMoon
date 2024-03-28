'''
Here we will use keras (https://keras.io/) to build our neural network.
'''
from .neuralnetdata import NeuralNetData
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler

class MyHyperParameters():
    """
    Class that defines data for input into the neural net

    Methods:
        None

    Attributes:
        nlyrs // Number of layers in the Neural Net
        npl // Number of nodes per layer in the form of a list
        lrate // Learning rate for the model
        batch_size // Number of entries (pixels) seen at a time
        nepochs // Number of times the model sees the training data
    """
    def __init__(self,nlyrs:'float',npl:'list',lrate:'float',batch_size:'float',nepochs:'float') -> None:
        self.nlyrs = nlyrs
        self.npl = npl
        self.lrate = lrate
        self.batch_size = batch_size
        self.nepochs = nepochs


def run_mlp(
        data:'NeuralNetData',
        hpms:'MyHyperParameters'
        ) -> 'np.ndarray':
    '''
    Here we run our labeled data through a simple multilayer perceptron
    '''

    model = keras.Sequential()

    model.add(keras.Input(shape=(data.X_train.shape[0],),name='Input_Layer'))

    for i in range(0,hpms.nlyrs):
        model.add(keras.layers.Dense(units=hpms.npl[i],activation=keras.activations.relu))
    
    model.add(keras.layers.Dense(units=data.num_labels))

    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=hpms.lrate),
        loss='mse'
    )

    history = model.fit(
    x=np.moveaxis(data.X_train,0,1),
    y=np.moveaxis(data.Y_train,0,1),
    batch_size=hpms.batch_size,
    epochs=hpms.nepochs,
    verbose='auto',
    validation_split=0.1,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    )

    return model