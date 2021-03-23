###############################################################################
# .. attention::
#
# Copyright(c) 2021 Francesco Loro, Master Degree Student Università degli studi di Padova.
# All rights reserved.
#
# This software component is licensed by Apache License Version 2.0
# http://www.apache.org/licenses/
# Same as QKeras
# You may not use this file except in compliance with# the License.
#
###############################################################################

__author__ = "Francesco Loro"
__email__ = "francesco.official@gmail.com"
__supervisor__ = "Danilo Pau"
__email__ = "danilo.pau@st.com"

# QKeras neural network zoo
#### Collection of pre-trained neural networks 
This folder contains a collection of networks written using two different frameworks: qkeras and larq. 
Each network can be built and tested using a randomly generated dataset, the output will consist of two measurements:
- Mean MSE, calculate the mean MSE between all the output for both networks
- Absolute errors, calculates how many times the class predicted by one network does
    not coincide with the class predicted by the other network

The folder is divided in:
- network_name.py is the class to build and test the networks.
- utils.py contains share methods between all the classes.
- results.txt are the results of the comparison between the networks.
- ./qkeras_models contains the .json qkeras saved models.
- ./larq_models contains the .json larq saved models.
- ./weights please put the downloaded weights here, link to download weights is provided in the network class.

Link to the folder with all weights: https://drive.google.com/drive/folders/1pGZ6dGWvJyc9aH-TOQohm0PhORihQZ5I?usp=sharing

An example can by run with:
```python 
 python3 quicknet.py