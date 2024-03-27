"""
Here we define a module for creating experimental data labels for our first round of neural network training. The essential framework will be to create labels based off of some user-specified threshold value in spectral parameter maps. The goal of of this inital process is not to classify spectra, but to validate the use of neural networks to learn about this spectral dataset.
"""
import numpy as np
from scipy.stats import wasserstein_distance

def spa_label(M:'np.ndarray',reference_spectrum:'np.ndarray',similarity_threshold:'float',shadowmap:'np.ndarray') -> 'np.ndarray':
    """
    We use this function to create labels based off of spectral angle mapping techniques. The spectral angle parameter (also known as the cosine distance) is defined by: 

        SPA = arccos((M . I) / (||M||*||I||))

    Where M is the input vector (i.e. one pixel of an image) and I is a reference spectrum. Lower SPA values indicate a higher level of similarity. The similarity threshold is measured in degrees.
    """

    M = np.moveaxis(M,[0,1,2],[2,0,1])

    total_pixels = M.shape[0]*M.shape[1]
    reference_spectrum = np.expand_dims(reference_spectrum,1)
    I = np.repeat(reference_spectrum,total_pixels,1).T
    I = I.reshape((M.shape))

    spa = 180*np.arccos(np.einsum('ijk,ijk->ij',M,I)/(np.linalg.norm(M,axis=2)*np.linalg.norm(I,axis=2)))/np.pi

    return spa<similarity_threshold

def emd_label(M:'np.ndarray',reference_spectrum:'np.ndarray',similarity_threshold:'float') -> 'np.ndarray':
    """
    We use this function to create labels based off of Wasserstein or Earth Mover's distance. This is done using the scipy.stats.wasserstein_distance function applied over an entire image.
    """

    vemd = np.vectorize()

    return None
