# ClassifyHySpecMoon

Welcome to ClassifyHySpecMoon by z-vig!

---
### Introduction

ClassifyHySpecMoon is a Python Package created for AOSC650 at the University of Maryland taught by Dr. Maria Molina. The goal of this package is to answer the following science question: 

> **What unique lithologies exist at the Gruithuisen Domes region of the Moon that may be able to tell us about their formation mechanism?**

To answer this question, a 1D Convolution neural network is ultimately developed to classify hyperspectral data from the Moon Mineralogy Mapper (M^3^) with the following specifications:

> - One mosaicked Image of the Gruithuisen Gamma and Northwest Domes
> - 238,272 Pixels (Samples)
> - 239 Spectral Bands (Features)

### Usage
To download the data, either download and mosaic the [images](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0) yourself (see below for image information), or request access for the [google drive link](https://drive.google.com/file/d/1kM4KgaWzTF3_yZFpEO19Nir1lTrWcW1f/view?usp=drive_link) (zvig@umd.edu).

> M^3^ Images Used: M3T20090418T020644 and M3T2009418T020848

To initialize the package, use python 3.11 and create a virtual environment fromthe requirements .txt file by running the following in a clean directory in Windows Powershell (or the equivalent for other OS).

`pip install .venv`
