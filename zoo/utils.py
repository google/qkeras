# Copyright 2021 Loro Francesco
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Francesco Loro"
__email__ = "francesco.official@gmail.com"
__supervisor__ = "Danilo Pau"
__email__ = "danilo.pau@st.com"

import numpy as np
from tqdm import tqdm
import os
import json
#import cv2

PATH_TO_LARQ = "./larq_models"
PATH_TO_QKERAS = "./qkeras_models"


def calculate_MSE(res_qkeras, res_larq):
    """
    Calculate the MSE between the two lists
    :param res_qkeras: list with the prediction given by the qkeras network
    :param res_larq: list with the prediction given by the larq network
    :return: the MSE
    """
    qres = np.squeeze(np.asarray(res_qkeras))
    lres = np.squeeze(np.asarray(res_larq))
    mse = (np.square(qres - lres)).mean()
    return mse


def calculate_absolute_error(res_qkeras, res_larq):
    """
    calculates how many times the class predicted by the qkeras network does
    not coincide with the class predicted by
    the larq network
    :param res_qkeras: list with the prediction given by the qkeras network
    :param res_larq: list with the prediction given by the larq network
    :return: number of misclassifications
    """
    pred = np.argmax(np.asarray(res_qkeras), axis=2)
    real = np.argmax(np.asarray(res_larq), axis=2)
    return np.count_nonzero(pred - real)


def create_random_dataset(sample_num=100):
    """
    Generate a random dataset to simulate imageNet
    :param sample_num: number of wanted samples, default is 100 samples
    :return: numpy array that represents the dataset
    """
    return np.random.randint(low=0, high=254, size=(sample_num, 1, 224, 224, 3))


def compare_network(qkeras_network, larq_network, dataset, network_name):
    """
    Given a shape dataset that conforms to the input of the networks.
    Compare the two input networks, based on the
    average MSE of their predictions and the number of times the two networks
    predict a different class.
    Prints the mean MSE and the Absolute error
    :param qkeras_network: qkeras network
    :param larq_network: larq network
    :param dataset: dataset
    :param network_name: network name
    """
    # list where predictions are stored
    res_qkeras = []
    res_larq = []

    for data in tqdm(dataset):
        res_qkeras.append(qkeras_network.predict(data))
        res_larq.append(larq_network.predict(data))
    mse = calculate_MSE(res_qkeras, res_larq)
    print("Mean MSE for",  network_name, "->", np.asarray(mse).mean())
    print("Absolute errors for",  network_name, "->",
          calculate_absolute_error(res_qkeras, res_larq))


def dump_network_to_json(qkeras_network, larq_network, network_name):
    """
    Dumps the given network to .json in the correct directory
    :param qkeras_network: qkeras network
    :param larq_network: larq network
    :param network_name: name of the network
    """
    network = qkeras_network.to_json()
    with open(os.path.join(PATH_TO_QKERAS, network_name + ".json"), "w") \
            as outfile:
        json.dump(network, outfile)
    print("Network:", network_name, "successfully saved into:",
          os.path.join(PATH_TO_QKERAS, network_name))

    network = larq_network.to_json()
    with open(os.path.join(PATH_TO_LARQ, network_name + ".json"), "w") \
            as outfile:
        json.dump(network, outfile)
    print("Network:", network_name, "successfully saved into:",
          os.path.join(PATH_TO_LARQ, str(network_name)))


def loadImageNetData(path, image_num=1000):
    """
    Load and preprocess images from the imageNet dataset, please specify the 
    directory where the dataset is
    and the desired number of samples
    :param path: path to the dataset
    :param image_num: number of wanted images
    :return: numpy array that contains the images with the correct shape for 
    the networks
    """
    file_list = os.listdir(path)
    file_list.sort()
    file_list = file_list[0:image_num]
    im = []
    for i in tqdm(file_list):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img, (224, 224))
        im.append(img)
        del img
    im = np.expand_dims(np.asarray(im), axis=1)
    return im