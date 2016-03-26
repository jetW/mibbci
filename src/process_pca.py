'''

    Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
        http://martinos.org/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html


'''

import params
import math
import threading
import numpy as np
import time
from datetime import datetime
import csv
from scipy import signal
from sklearn.decomposition import PCA
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
    label_feed = data_loaded[:, 17:20]
    print 'X_raw.shape', X_raw.shape


    # Preprocess the raw data
    print "X_raw[:, 1].shape:", X_raw[:, 1].shape
    X_preproc = X_raw - np.tile(np.reshape(X_raw[:, 1], (X_raw.shape[0], 1)), [1, params.NUM_CHANNELS]);


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
    X_tdfilt = signal.lfilter(numer, denom, X_preproc.T).T


    # Get the signal power
    X_pow = X_tdfilt * X_tdfilt


    # Get moving average on signal power
    X_ma = signal.convolve(X_pow.T, np.ones((1, int(1*params.FREQ_S)))).T


    # Get channel differences
    X_to_diff = X_ma
    X_diff_c5_c6 = X_to_diff[:, 3] - X_to_diff[:, 9]
    X_diff_c3_c4 = X_to_diff[:, 4] - X_to_diff[:, 8]
    X_diff_c1_c2 = X_to_diff[:, 5] - X_to_diff[:, 7]
    X_diff_fc3_fc4 = X_to_diff[:, 0] - X_to_diff[:, 2]
    X_diff_cp5_cp6 = X_to_diff[:, 10] - X_to_diff[:, 12]
    X_diff_p3_p4 = X_to_diff[:, 13] - X_to_diff[:, 15]
    X_diff_avg = (X_diff_c5_c6 + X_diff_c3_c4 + X_diff_c1_c2 + X_diff_fc3_fc4 + X_diff_cp5_cp6 + X_diff_p3_p4) / 6.0


    # Plot the signal
    if True:
        channels_to_plot = [1, 3, 9]
        #plt.plot(X_tdfilt[:, channels_to_plot])
        #plt.plot(X_ma[:, channels_to_plot])
        plt.plot(X_diff_c1_c2)
        plt.plot(X_diff_c3_c4)
        plt.plot(X_diff_c5_c6)
        plt.plot(X_diff_fc3_fc4)
        plt.plot(X_diff_cp5_cp6)
        plt.plot(X_diff_p3_p4)
        #plt.plot(X_diff_avg)
        plt.plot(1000*label_feed[:, 0:2])
        plt.legend(['X_diff_c1_c2', 'X_diff_c3_c4', 'X_diff_c5_c6', 'X_diff_fc3_fc4', 'X_diff_cp5_cp6', 'X_diff_p3_p4', 'rh', 'lh'])
        plt.xlim([1500, 5000])
        plt.ylim([-2000, 2000])
        plt.show()
        #X_raw_mne.plot(events=events_mne, event_color={1: 'cyan'})
        #X_raw_mne.plot(events=event_series, event_color={1: 'cyan', -1: 'lightgray'})
        #time.sleep(2) no


