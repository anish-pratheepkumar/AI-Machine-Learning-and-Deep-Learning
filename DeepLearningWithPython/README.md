# Deep Learning with Python 

This section contains implementation of exercises mentioned in the book Deep Learning with Python by François Chollet. The implementation is based on the KERAS framework.It provides insight into understanding basic concepts of implementing a NeuralNetwork, training, analysing, regularisation etc.

### [01_BostonPriceRegression](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/01_BostonPriceRegression)

This task consists of predicting the median price of homes in a Boston suburb. It has relatively few data points: only 506, split between 404 training samples and 102 test samples. Here we implement a K-fold validation to tackle regression problems with less data points.

### [02_IMDBbinClassification](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/02_IMDBbinClassification)

This is a binary classidfication problem in which a set of 50,000 highly polarised reviews from IMDB are used. The network is trained to predict if the review is positive or negative. a Fully Connected NN is used here with the input reviews converted into a binary tensor by one-hot encoding. Importance of validation dataset and stopping the training to avoid overfitting can also be observed.


### [03_ReutersMulticls Classification](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/03_ReutersMulticlsClassification)

The Multi Class classification is implemented here. Reuters dataset with 8982 training samples and 2246 test samples. The News wires are classified into 46 different topics using a Fully Connected Neural Network. The data set is one-hot encoded before feeding into the network.

### [04_MnistClassification](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/04_MnistClassification)

The MNIST data is classified using a Fully Connected Network.

### [05_Intro2Conv](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/05_Intro2Conv)

The 'Hello World!' of NN - MNIST dataset is used to train a CNN network to understand the basic structure of a CNN.

### [06_CatsVsDogs](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs)

* [CatsVsDogs-CNN](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/CatsVsDogs-CNN.ipynb)

Here from a total of 4000 cats and dogs images a CNN is implemented to classify them correctly. Followed by that two regularisation techniques namely Data Augmentation and Dropout are implemented to improve the performance of the network.

* [CatsVsDogs-PretrainedCNN](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/CatsVsDogs-PretrainedCNN.ipynb)

Now a pretrained CNN - VGG16 is used on the same dataset and evaluated its performance.

* [CatsVsDogs-PretrainedCNN-withAugmentation_Colab](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/CatsVsDogs-PretrainedCNN-withAugmentation_Colab.ipynb)

Few base layers of pretrained VGG16 is set trainable and then the network is trained with the same input. Also the input data is subjected to regularisation - Augmentation. The accuracy of the network increases with this.

* [VisualiseCAM](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/VisualiseCAM.ipynb)

The Class Activation(output) Map (CAM) is visualised in this file. It gives insight into which part of a given image lead to its final classification decision by producing heat maps of class activation over input images.

* [VisualiseConvnetFilter-info -40iteration](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/VisualiseConvnetFilter-info%20-40iteration.ipynb)

CNN filter visualisations tell a lot about how each layer sees its input images. This is done by displaying the visual pattern that each filter is meant to respond to.

* [VisualiseInterActivations](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/blob/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/06_CatsVsDogs/VisualiseInterActivations.ipynb)

The activations of intermediate layers as per requirement is visualised in this file. This can help in understanding how different convolutional layers modify their input and to det an idea of what each layer does. An interesting plot of filter activations can be seen in this notebook.


### [07_KerasCallbacks](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/07_KerasCallbacks)

The callback function of early stopping is implemented here. The training will be stopped if the validation loss doesnot improve and hence preventing overfitting.


### [08_TensorBoard](https://github.com/anish-pratheepkumar/GitDeepLearningStudy/tree/master/DeepLearningWithPython-Fran%C3%A7ois%20Chollet/08_TensorBoard)

The TensorBoard callback is used to visually monitor the training as it is going on. The code for the same is implemented for training a MNIST data set.

