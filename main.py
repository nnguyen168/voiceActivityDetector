#!usr/bin/python
# -*- coding: utf-8 -*-

####################################################################################################################
__author__ = 'Nam NGUYEN HOAI <nguyenhoainam2k11@gmail.com>'											   		   #
__date__, __version__ = '28/10/2017', '1.0'														   				   #
__description__ = u'''This script builds the neural networks for the Voice Activity Detector (VAD)'''              #
####################################################################################################################

import os
import itertools
from data_preprocessing import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    # Prepare the data
    file_list       = [f for f in os.listdir('/Users/namnguyen/Self-taught/Snips/vad_data') if f.endswith(".wav")] # get the name of .wav files in the directory
    annotation_list = [f for f in os.listdir('/Users/namnguyen/Self-taught/Snips/vad_data') if f.endswith(".json")] # get the name of .json file in the directory
    nperseg         = 512
    nfft            = 512
    data_tot        = np.array([])
    labels_tot      = []
    for i in range(len(file_list[:50])):
        print file_list[i]
        m_spectro, time, freq   = createSpectro('vad_data/'+file_list[i], nperseg, nfft)
        annotation_time, labels = timeAnnotation('vad_data/'+annotation_list[i], time)
        annotation_index        = indexAnnotation(annotation_time,time)
        data                    = extractData(m_spectro, annotation_index, freq)
        data_tot                = np.concatenate((data_tot,data),axis=0)
        labels_tot              = labels_tot + labels
    '''
    wav_file                = 'vad_data/19-198-0003.wav'
    nperseg                 = 512
    nfft                    = 512
    m_spectro, time, freq   = createSpectro(wav_file, nperseg, nfft)
    annotation_time, labels = timeAnnotation('vad_data/19-198-0003.json', time)

    annotation_index        = indexAnnotation(annotation_time,time)
    data                    = extractData(m_spectro, annotation_index, freq)
    max_length              = max([len(i) for i in data])
    for i in range(len(data)):
        data[i]     = np.concatenate((data[i],np.zeros(max_length-len(data[i]))), axis=0)
    #data                    = np.asarray(data)

    # Balance between voiced and unvoiced data
    data_voiced             = []
    data_unvoiced           = []
    for i in range(len(labels)):
        if labels[i] == 1: #voiced
            data_voiced.append(data[i])
        else: #unvoiced
            data_unvoiced.append(data[i])

    data_length             = min(len(data_unvoiced), len(data_voiced))
    data_voiced             = np.asarray(data_voiced[:data_length])
    data_unvoiced           = np.asarray(data_unvoiced[:data_length])
    data                    = np.concatenate((data_voiced,data_unvoiced), axis=0)
    labels                  = np.concatenate((np.ones(data_length), np.zeros(data_length)))
    '''

    max_length              = max([len(i) for i in data_tot])
    for i in range(len(data_tot)):
        data_tot[i]         = np.concatenate((data_tot[i], np.zeros(max_length-len(data_tot[i]))), axis=0) # zero padding

    # Balance between voiced and unvoiced data
    data_voiced             = []
    data_unvoiced           = []
    for i in range(len(labels_tot)):
        if labels_tot[i] == 1: #voiced
            data_voiced.append(data_tot[i])
        else: #unvoiced
            data_unvoiced.append(data_tot[i])

    data_length             = min(len(data_unvoiced), len(data_voiced))
    data_voiced             = np.asarray(data_voiced[:data_length])
    data_unvoiced           = np.asarray(data_unvoiced[:data_length])
    data                    = np.concatenate((data_voiced,data_unvoiced), axis=0)
    labels                  = np.concatenate((np.ones(data_length), np.zeros(data_length)))

    # One hot encoding the labels
    onehotencoder           = OneHotEncoder(categorical_features ='all')
    labels_encoded          = onehotencoder.fit_transform(np.asarray(labels).reshape(-1,1)).toarray()

    training_ratio          = 0.75
    training_index          = int(data_length*training_ratio)
    training_data, test_data    = np.concatenate((data_voiced[:training_index], data_unvoiced[:training_index])),np.concatenate((data_voiced[training_index:], data_unvoiced[training_index:]))

    # Build the neural networks
    # The autoencoder
    encoding_dim            = 200
    input_dim               = Input(shape=(max_length,))
    encoded                 = Dense(encoding_dim, activation='relu')(input_dim)
    decoded                 = Dense(max_length, activation='sigmoid')(encoded)
    autoencoder             = Model(input_dim,decoded)

    #encoder                 = Model(input_dim, encoded)
    encoder_layer           = autoencoder.layers[1]
    encoder                 = Model(input_dim, encoder_layer(input_dim))
    encoded_input           = Input(shape=(encoding_dim,))
    decoder_layer           = autoencoder.layers[-1]
    decoder                 = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(training_data, training_data, epochs=50, batch_size=128,
                    shuffle=True, validation_data=(test_data,test_data))

    encoded_data_voiced     = encoder.predict(data_voiced) # get the compressed data
    encoded_data_unvoiced   = encoder.predict(data_unvoiced)
    labels_encoded_voiced   = labels_encoded[:data_length]
    labels_encoded_unvoiced = labels_encoded[data_length:]

    # The binary classification
    #labels                  = np.asarray(labels).astype('float32')
    training_data, test_data        = np.concatenate((encoded_data_voiced[:training_index], encoded_data_unvoiced[:training_index])),np.concatenate((encoded_data_voiced[training_index:], encoded_data_unvoiced[training_index:]))
    training_labels, test_labels    = np.concatenate((labels_encoded_voiced[:training_index], labels_encoded_unvoiced[:training_index])),np.concatenate((labels_encoded_voiced[training_index:], labels_encoded_unvoiced[training_index:]))

    model                   = Sequential()
    model.add(Dense(16, activation='relu', input_dim=200))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_data, training_labels, epochs=50,
                batch_size=16, validation_data=(test_data, test_labels))

    predict_labels          = model.predict(test_data)
    predict_labels          = np.argmax(predict_labels, axis=1)
    cnf_matrix              = confusion_matrix(np.argmax(test_labels, axis=1), predict_labels)
    class_names             = np.array(['unvoiced', 'voiced'])
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()
    #plt.savefig('confusion_matrix.png')
