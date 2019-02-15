"""
This python script generates noisy samples and stores it in a pickle dump
"""

import numpy as np
from LoadMNIST import mnist_fashion
import matplotlib.pyplot as plt
import pickle

noise_probability = 0.8
noTrPerClass = 550
noValPerClass = 50
noTsPerClass = 100
noTrSamples= 6000
noTsSamples= 1000
digit_range = [0,1,2,3,4,5,6,7,8,9]
output_file  = "fashion_train_validation_test_split_data"+\
                "_noTr_"+str(noTrSamples)+\
                "_noTs_"+str(noTsSamples)+\
                "_digits_"+str(digit_range)+\
                "_noise_"+str(noise_probability)
output_file = output_file.replace(".","_")
output_file += ".txt"


train_data, train_label, test_data, test_label = \
        mnist_fashion(noTrSamples=noTrSamples,noTsSamples=noTsSamples,\
        digit_range=digit_range,\
        noTrPerClass=noTrPerClass+noValPerClass, noTsPerClass=noTsPerClass)

print("loading data from mnist done")
print("now adding noise")

## Code to create validation data
train_labels = train_label.reshape((train_label.shape[1]))




noisy_train_data = np.array(train_data)
noisy_test_data = np.array(test_data)

for i in range(noisy_train_data.shape[1]):
    for j in range(noisy_train_data.shape[0]):
        r = np.random.rand(1)
        if r < noise_probability :
            noisy_train_data[j, i] = np.random.rand(1)

for i in range(noisy_test_data.shape[1]):
    for j in range(noisy_test_data.shape[0]):
        r = np.random.rand(1)
        if r < noise_probability :
            noisy_test_data[j, i] = np.random.rand(1)
            
print("noise adding done")
            
            
idVal = []
idTr = []

for ll in digit_range:
    idx = np.where(train_labels == ll)
    #print(idx)
    for ii in idx[0][noTrPerClass: ]:
        idVal.append(ii)
    for ii in idx[0][ : noTrPerClass]:
        idTr.append(ii)
    
idVal = np.array(idVal)
idTr = np.array(idTr)

print("index gathering done")



"""
idx = np.where(train_labels == digit_range[0])
#print("idx.shape = ", idx[0].shape)
idTr = idx[0][ : noTrPerClass] # contains the column numbers to be contained in the training dataset
idVal = idx[0][noTrPerClass : ] #contains column numbers to be included in validation data 
"""
val_noisyX = noisy_train_data[ : , idVal]
val_originalX = train_data[ : , idVal]
valY = train_label[ : , idVal]
tr_noisyX = noisy_train_data[: , idTr]
tr_originalX = train_data[: , idTr]
trY = train_label[:, idTr]

"""
count = 0
for ll in digit_range[1:]:
    print("ll = ", ll)
    idx = np.where(train_labels == ll)
    idTr = idx[0][ : noTrPerClass] # contains the column numbers to be contained in the training dataset
    idVal = idx[0][noTrPerClass : ] #contains column numbers to be included in validation data 
    val_noisyX = np.vstack((val_noisyX.T, noisy_train_data[ : , idVal].T)).T
    val_originalX = np.vstack((val_originalX.T, train_data[ : , idVal].T)).T
    valY = np.vstack((valY.T, train_label[ : , idVal].T)).T
    tr_noisyX = np.vstack((tr_noisyX.T, noisy_train_data[ : , idTr].T)).T
    tr_originalX = np.vstack((tr_originalX.T, train_data[ : , idTr].T)).T
    trY = np.vstack((trY.T, train_label[ : , idTr].T)).T
 """   
output = {}
output["train_data"] = tr_originalX
output["noisy_train_data"] = tr_noisyX
output["train_label"] = trY
output["validation_data"] = val_originalX
output["noisy_validation_data"] = val_noisyX
output["validation_label"] = valY
output["test_data"] = test_data
output["noisy_test_data"] = noisy_test_data
output["test_label"] = test_label

with open(output_file, "wb") as fp:
    pickle.dump(output, fp)
