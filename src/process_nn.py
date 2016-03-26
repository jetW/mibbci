

import params
import math
import threading
import numpy as np
import time
from datetime import datetime
import csv
from scipy import signal
from joblib import Parallel, delayed
import matplotlib.pyplot as plt




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'


    # Load the data
    data_filename = '../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv'
    data_loaded = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
    print 'data_loaded.shape:', data_loaded.shape
    X_raw = data_loaded[:, 1:(1+params.NUM_CHANNELS)]
    time_axis = data_loaded[:, 0]
    label_feed = data_loaded[:, 17:21]
    X_feat = data_loaded[:, 23]
    print 'X_raw.shape', X_raw.shape

    # Preprocess the raw data
    X_preproc = X_raw

    # Initialize the time-domain filter
    freq_Nyq = params.FREQ_S/2.
    freq_trans = 0.5
    freqs_FIR_Hz = np.array([8.-freq_trans, 12.+freq_trans])
    #numer = signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = signal.firwin(params.M_FIR, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = 1.
    '''w, h = signal.freqz(numer)
    plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()'''

    # Filter in time domain
    #X_tdfilt = signal.lfilter(numer, denom, X_preproc.T).T
    X_tdfilt = X_preproc
