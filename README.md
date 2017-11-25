# Voice Activity Detector
A simple voice activity detector by neural networks
## Getting started
This is a simple project to carry out the voice activity detector (VAD) task using an architecture of type neural networks. The implementation will be conducted by Python
## Prerequisites
Python librairies need to be installed before the implementation
* Numpy
```
pip install numpy
```
* Scipy
```
pip install scipy
```
* Matplotlib
* Keras
https://keras.io
* Sklearn
```
pip install -U scikit-learn
```
## Running the codes
### Data preprocessing
* _Data provided_: speech .wav files along with the speech segments as start times and end times
* _Spectrogram_: build the spectrogram of the speech based on the Short time Fourier transform. Below is an example of the spectrogram, with the time varies from 0 to 15 seconds and the frequency varies from 0 to 8000 Hz.
![Spectrogram](https://github.com/BornToWin/voiceActivityDetector/blob/master/spectrogram.jpg)
* _Extract the data_: based on the annotation about the speech segments, the voiced and un-voiced parts of the speech were extracted from the spectrogram. For the frequency, a range from 300 to 3000Hz was used as this is the common range for the human voice.
### Algorithm
* The principal is to treat the extracted segments as images and to use the neural network to classify those segments.
* After extraction, the segments were transformed into a 1-D array, the values were normalized between 0 and 1, and the zero padding was used to get all the arrays with the same length.
* An architecture of type sparse auto encoder (SAE) was used to extract the features from the spectrogram.  The key of a is that by using the number of units in hidden layers smaller than the input dimension, then reconstruct the input at the output, we would be able to decrease the dimension of the input without losing information.
* A simple architecture was used as a binary classifier using neural networks to classify between two classes which are voiced and unvoiced segment.
### Implementation
![Neural network architecture](https://github.com/BornToWin/voiceActivityDetector/blob/master/NNarchitecture.jpg)
As can be seen from the architecture, the first part of the VAD is feature extraction by using a Sparse Auto Encoder (SAE). By putting n hidden units in the hidden layer of the SAE, we can decrease the dimension of the input segments.
After training the SAE, the output of the hidden layers was feed into a feed forward neural network which has two hidden layers, each has 16 hidden units. The output of the feedforward neural network has 2 dimension corresponding to the 2 classes need to be classified Voiced and Unvoiced.
### Result
The VAD was implemented on a data set of 1118 segments with the dimension of 72326 (including zero padding). -	The SAE which has the hidden layer with 200 units was trained on 50 iterations with the batch size of 128. After 50 epochs, the loss is __0.2018__. The binary classifier has two hidden layers 16 units each was trained on 50 iterations with the batch size of 16.

![Confusion matrix](https://github.com/BornToWin/voiceActivityDetector/blob/master/confusionMatrix.png) 

As can be seen in the confusion matrix, 106 unvoiced segments were classified correctly and 34 unvoiced segments were classified as voiced segments. 101 voiced segments were classified correctly and 39 were classified as unvoiced segments.
### Conclusion
With a False alarm (FA) of 25.2% and the miss probability of 27%, this VAD architecture shows the possibility of using neural networks as a binary classifier but still needs more modification for accurate use
