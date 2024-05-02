'''
Here we will use keras (https://keras.io/) to build our neural network.
'''
from .neuralnetdata import NeuralNetData
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

class MyHyperParameters():
    """
    Class that defines data for input into the neural net

    Methods:
        None

    Attributes:
        nlyrs // Number of layers in the Neural Net
        npl // Number of nodes per layer in the form of a list
        apl // activation for each layer as a list of strings
        lrate // Learning rate for the model
        batch_size // Number of entries (pixels) seen at a time
        nepochs // Number of times the model sees the training data
    """
    def __init__(self,nlyrs:'float',npl:'list',apl:'list',lrate:'float',batch_size:'float',nepochs:'float') -> None:
        self.nlyrs = nlyrs
        self.npl = npl
        self.apl = apl
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

    model.add(keras.Input(shape=(data.X_train.shape[1]),name='Input_Layer'))

    for i in range(0,hpms.nlyrs):
        model.add(
            keras.layers.Dense(
                units=hpms.npl[i],
                activation=keras.activations.relu
            )
        )
    
    model.add(keras.layers.Dense(units=data.Y.shape[1],activation="softmax"))

    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=hpms.lrate),
        loss='binary_crossentropy'
    )

    print(model.summary())

    train_decoded = np.argmax(data.Y_train,axis=1)

    history = model.fit(
    x=data.X_train,
    y=data.Y_train,
    batch_size=hpms.batch_size,
    epochs=hpms.nepochs,
    verbose='auto',
    validation_split=0.1,
    validation_data=None,
    shuffle=True,
    class_weight = dict(
        enumerate(
            compute_class_weight('balanced',
                                 classes=np.unique(train_decoded),
                                 y=train_decoded))),
    sample_weight=None,
    initial_epoch=0
    )

    predictions = model.predict(
        x = data.X_test
    )

    return model,history,predictions

def get_model(hpms:'MyHyperParameters'):
    '''
    Here we run our labeled data through a simple multilayer perceptron
    '''

    model = keras.Sequential()

    model.add(keras.Input(shape=(239)))

    for i in range(0,hpms.nlyrs):
        model.add(keras.layers.Dense(units=hpms.npl[i],activation=keras.activations.relu))
    
    model.add(keras.layers.Dense(units=7,activation="softmax")) #SOFTMAX WAS THE KEY!!

    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=hpms.lrate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

def get_conv_model(hpms:'MyHyperParameters'):
    '''
    Here we run our labeled data through one convolutional layer and then through a simple multilayer perceptron
    '''

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(239,1)))

    model.add(keras.layers.Conv1D(
        filters = 10,
        kernel_size = 30,
        strides = 1,
        padding = "same",
        input_shape = (239,1) #roughly matching a 1 micron band size
    ))
    model.add(keras.layers.MaxPooling1D(
        pool_size=2
    ))
    
    model.add(keras.layers.Conv1D(
        filters = 10,
        kernel_size = 15,
        strides = 1,
    input_shape = (239,1) #roughly matching a 1 micron band size
    ))

    model.add(keras.layers.MaxPooling1D(
        pool_size=2
    ))

    model.add(keras.layers.Flatten())

    for i in range(0,hpms.nlyrs):
        model.add(keras.layers.Dense(units=hpms.npl[i],activation=keras.activations.relu))
    
    model.add(keras.layers.Dense(units=7,activation="softmax")) #SOFTMAX WAS THE KEY!!

    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=hpms.lrate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

def tune_model(hp):
    nl = hp.Int("n_layers",
                   min_value = 1,
                   max_value = 4,
                   default = 2
                   )
    
    npl = [hp.Int(f"units_{i}",
                  min_value=10,
                  max_value=200,
                  step=10)
          for i in range(nl)]
    
    apl = [hp.Choice(f"activation_{i}",
                     ["relu","tanh"]) 
          for i in range(nl)]
    
    lr = hp.Float("learning_rate",
                  min_value=0.001,
                  max_value=10,
                  step=10,
                  sampling="log")
    
    bs = hp.Int("batch_size",
                min_value=500,
                max_value=1500,
                step=100)
    
    # ne = hp.Int("nepochs",
    #             min_value=10,
    #             max_value=100,
    #             step=10)
    
    hyperparams = MyHyperParameters(
        nlyrs = nl,
        npl = npl,
        apl=apl,
        lrate=lr,
        batch_size=bs,
        nepochs=20
    )

    model = get_conv_model(hyperparams)

    print(model.summary())

    return model