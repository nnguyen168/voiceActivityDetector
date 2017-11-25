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
