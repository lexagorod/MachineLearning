# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:59:36 2018

@author: Aleksei
"""

import network, network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#training_data = list(training_data)
#print(training_data[0:2])
#n = len(training_data)
#mini_batches = [
#                training_data[k:k+10]
    #            for k in range(0, n, 10)]
#print(training_data[0:15])
net = network.Network([784, 5, 10])
net.SGD(training_data, 30, 4, 3.0, test_data=test_data)
"""
net = network2.Network([784,30,10], cost =network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 40, 10, 0.5, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
"""