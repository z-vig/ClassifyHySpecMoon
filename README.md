# AOSC654 Term Project: ClassifyHySpecMoon

Welcome to ClassifyHySpecMoon by z-vig!

---
### Introduction

`ClassifyHySpecMoon` is a Python Package created for AOSC650 at the University of Maryland taught by Dr. Maria Molina. The goal of this package is to answer the following science question: 

> **What unique lithologies exist at the Gruithuisen Domes region of the Moon that may be able to tell us about their formation mechanism?**

To answer this question, a 1D Convolution neural network is ultimately developed to classify hyperspectral data from the Moon Mineralogy Mapper (M^3^) with the following specifications:

> - One mosaicked Image of the Gruithuisen Gamma and Northwest Domes
> - 238,272 Pixels (Samples)
> - 239 Spectral Bands (Features)

### Installation
To download the data, either download and mosaic the [images](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0) yourself (see below for image information), or request access for the [google drive link](https://drive.google.com/file/d/1kM4KgaWzTF3_yZFpEO19Nir1lTrWcW1f/view?usp=drive_link) (zvig@umd.edu).

> M^3^ Images Used: M3T20090418T020644 and M3T2009418T020848

To initialize the package, use python 3.11, clone the github repo and create a virtual environment from the requirements.txt file by running the following in a clean directory in Windows Powershell (or the equivalent for other OS).

`git clone https://github.com/z-vig/ClassifyHySpecMoon.git`

This clones the github repo to your current directory.

`{PATH_TO}/python.exe -m venv .venv`

This creates a new directory called .venv iun your current directory where your virtual environemnt will live. To switch to this environment and install packages:

`.venv/Scripts/Activate.ps1`

At this point, you may run into an error in which powershell will not allow you to run scripts on your machine. To remedy this, run an instance of powershell as administrator (right-click powershell &rarr; run as administrator) and run:

`Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope LocalMachine`

To learn more about execution policies and the potential dangers of setting your execution policy to unrestricted, [see here](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4).

`{PATH_TO}/python.exe -m pip install -r requirements.txt`

This will finally install all the requirement python packages at their required versions for ClassifyHySpecMoon, given you are installing them with python 3.11. Now, all that is left to do is to perform a local import in the same github repo directory.

```python
import classifyhyspecmoon
```

### Usage
`ClassifyHySpecMoon` utilizes two main classes and one submodule to build a neural network using `keras_tuner` and `Scikit-Learn`.

#### `NeuralNetData` Class
This class allows the easy input and preprocessing of data in an HDF5 format. To initialize an object, simply call

```python
from classifyhyspecmoon.neuralnetdata import NeuralNetData 
myNNdata = NeuralNetData(path\to\data.hdf5)
```

From here, utilize the methods documented in the package docstrings to perform standard preprocessing steps defined in `scikit-learn` and `keras` using simply one-line coding steps. The use of this class allows for a clean and easy to follow user interface in the form of a jupyter notebook (see below).

#### `MyHyperParameters` Class
Although `keras` has a built-in class for handling hyperparameters, this class allows the user to modify which hyperparameters they want to focus on, which allows a simple interface with `keras_tuner`. Simply call the class and add hyperparameters as follows:

```python
from classifyhyspecmoon.buildNN import MyHyperParameters
myhp = MyHyperparameters(
    nlyrs = float,
    npl = list,
    apl = list,
    lrate = float,
    batch_size = float,
    nepochs = float
)
```
The documentation for the definition of these parameters are in the docstrings. These are currently the only hyperparameters available, but more may be added in the future.

#### `run_mlp()` and `tune_model()` functions
These two functions are what actually allow the user to build a neural network using `keras`. `run_mlp()` allows the user to run a simple dense multilayer perceptron, whereas the `tune_model()` function can be input to a keras tuner object as follows:

```python
tuner = keras_tuner.RandomSearch(
    hypermodel=tune_model,
    objective='val_loss',
    max_trials=5,
    overwrite=True,
    project_name='aosc_project'
)
```

#### Jupyter Notebooks
Including in the package are three jupyter notebooks showing the progression of building my neural network for the AOSC654 term project. The order in which they were developed is as follows:

1. `regression_model.ipynb`
2. `one_hot_model.ipynb`
3. `tuned_CNN_model.ipynb`

These notebooks start with a basic regression-style multi-layer perceptron and progress by one-hot encoding and adding class weights to finally adding in 1D convolutional layers and model asessment capabilities. The first notebook may not currently run due to deprecated capabilities in `NeuralNetData`, but the final model in `tuned_CNN_model.ipynb` should be working just fine.

Thanks so much for trying out my work, and I hope you enjoy!
