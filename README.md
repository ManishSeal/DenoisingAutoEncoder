# DenoisingAutoEncoders

This project is a demonstration of the de-noising capability of an Auto-Encoder.

Here we use the FASHION-MNIST data set as is available on: https://github.com/zalandoresearch/fashion-mnist. The noise added is salt and pepper noise, the amount of noise added can be controlled inside the CreateNoisyData.py

To run:
1) Download the Fashion-MNIST dataset from the link above
2) In the LoadMNIST.py file provide a file path to the folder containing unzipped data files
3) Run the CreateNoisyData.py to generate noisy version of the Fashion-MNIST dataset
4) Under the main function for the DenoisingAutoEncoder.py update the file paths, the dimension of the network, number of epochs and the learning rate.


Note: The model is a very simple model. It is supposed to be a single layer stacked encoder, putting higher dimensions for the net_dims might result in unexpected results or failures. The learning rate is kept constant. No momentum is used. Batch Gradient Descent is used.
