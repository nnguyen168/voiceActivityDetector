#!usr/bin/python
# -*- coding: utf-8 -*-

####################################################################################################################
__author__ = 'Nam NGUYEN HOAI <nguyenhoainam2k11@gmail.com>'											   		   #
__date__, __version__ = '28/10/2017', '1.0'														   				   #
__description__ = u'''This script prepares the data for the Voice Activity Detector (VAD) using neural networks''' #
####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.io import wavfile
from scipy import signal
import warnings
warnings.filterwarnings("error")

def createSpectro(wav_file, NPERSEG, NFFT):
    '''
    Return the spectrogram of the audio data
    '''
    rate, data  = wavfile.read(wav_file)
    f,t,Zxx     = signal.stft(data, rate, window='hamming', nperseg=NPERSEG, nfft=NFFT) # calculate the short time fourrier transform
    try:
        m_spectro   = 20*np.log10(np.abs(Zxx))
    except RuntimeWarning:
        m_spectro   = 20*np.abs(Zxx)
    t           = np.around(t, decimals=2)
    return m_spectro, t, f

def timeAnnotation(annotation_file, time):
    '''
    Get the annotation time from the json file
    '''
    labels          = []
    with open(annotation_file) as data_file:
        data = json.load(data_file)
    speech_time     = []
    for p in data['speech_segments']:
        speech_time.append([round(p['start_time'],2),round(p['end_time'],2)])
    start_file      = time[0]
    end_file        = time[-1]
    annotation_time = []
    for i in range(len(speech_time)):
        if i == 0:
            if speech_time[i][0] != start_file:
                labels.append(0)
                annotation_time.append([start_file, round(speech_time[i][0] - start_file,2)])
        else:
            if speech_time[i][0] != speech_time[i-1][1]:
                labels.append(0)
                annotation_time.append([speech_time[i-1][1],speech_time[i][0]])
        annotation_time.append(speech_time[i])
        labels.append(1)
    if annotation_time[-1][-1] != end_file:
        annotation_time.append([annotation_time[-1][-1],end_file])
        labels.append(0)
    return annotation_time, labels

def indexAnnotation(annotation_time, time):
    '''
    Get the segment index from the spectrogram based on the annotation time
    '''
    annotation_index    = []
    for p in annotation_time:
        start_time  = p[0]
        end_time    = p[1]
        min_start   = np.inf
        min_end     = np.inf
        for i in range(len(time)):
            if np.abs(time[i]-start_time) < min_start:
                c1          = i
                min_start   = np.abs(time[c1]-start_time)
            if np.abs(time[i]-end_time) < min_end:
                c2          = i
                min_end     = np.abs(time[c2]-end_time)
        annotation_index.append([c1,c2])
    return annotation_index

def normalize(array):
    '''
    Normalize a 1-D array to have values between 0 and 1
    '''
    max_array   = np.max(array, axis=0)
    min_array   = np.min(array, axis=0)
    return (array-min_array)/(max_array-min_array)

def extractData(m_spectro, annotation_index, freq):
    '''
    Extract data from the spectrogram
    '''
    data        = []
    start_freq  = 300
    end_freq    = 3000
    min_start   = np.inf
    min_end     = np.inf
    for i in range(len(freq)):
        if np.abs(freq[i]-start_freq) < min_start:
            r1          = i
            min_start   = np.abs(freq[r1]-start_freq)
        if np.abs(freq[i]-end_freq) < min_end:
            r2          = i
            min_end     = np.abs(freq[r2]-end_freq)
    for i in annotation_index:
        data.append(m_spectro[r1:r2,i[0]:i[1]+1])
    #length      = []
    for i in range(len(data)):
        '''
        for j in range(len(data[i])):
            data[i][j]         = data[i][j]/np.max(np.abs(data[i][j]), axis=0)
        '''
        data[i]         = data[i].flatten()
        data[i]         = normalize(data[i])
        #length.append(len(data[i]))
    #max_len     = max(length)
    #for i in range(len(data)):
        #data[i]     = np.concatenate((data[i],np.zeros(max_len-len(data[i]))), axis=0)
    return data

if __name__ == '__main__':
    wav_file                = 'vad_data/19-198-0003.wav'
    nperseg                 = 512
    nfft                    = 512
    m_spectro, time, freq   = createSpectro(wav_file, nperseg, nfft)
    annotation_time, labels = timeAnnotation('vad_data/19-198-0003.json', time)
    annotation_index = indexAnnotation(annotation_time,time)
    data    = extractData(m_spectro, annotation_index, freq)
    print len(data)
    print data[0]
