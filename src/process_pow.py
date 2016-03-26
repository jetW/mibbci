'''

    Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
        http://martinos.org/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html


'''

import params
import mne
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
    #data_filename = 'BME_BCI_MI_REC_20160310_09h00m52s.csv'
    #data_filename = 'BME_BCI_MI_REC_20160310_09h02m40s.csv'
    data_filename = 'BME_BCI_MI_REC_20160310_09h03m24s.csv'
    #data_filename = 'BME_BCI_MI_REC_20160310_09h04m05s.csv'
    data_loaded = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
    print 'data_loaded.shape:', data_loaded.shape
    X_raw = data_loaded[:, 1:(1+params.NUM_CHANNELS)]
    time_axis = data_loaded[:, 0]
    label_feed = data_loaded[:, 17:21]
    X_feat = data_loaded[:, 23]
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


    # Epoch the data
    X_to_epoch = X_tdfilt
    montage = mne.channels.read_montage('standard_1005', params.CHANNEL_NAMES)
    X_info = mne.create_info(params.CHANNEL_NAMES, params.FREQ_S, ch_types='eeg', montage=montage)
    X_mne = mne.io.RawArray(X_to_epoch.T, info=X_info, verbose=None)
    t_min = 0
    t_max = 1
    event_name_rh = 'rh'
    event_id_rh = 0
    event_series_rh = np.reshape(label_feed[:, event_id_rh], (label_feed.shape[0], 1))
    #print 'event_series.shape:', event_series.shape
    events_info = mne.create_info([event_name_rh], params.FREQ_S, ch_types='eeg', montage=None)
    events_raw_mne = mne.io.RawArray(event_series_rh.T, info=events_info, verbose=None)
    events_rh_mne = mne.find_events(events_raw_mne, stim_channel=event_name_rh, verbose=None)
    #print 'events_rh_mne', events_rh_mne
    event_name_lh = 'lh'
    event_id_lh = 0
    event_series_lh = np.reshape(label_feed[:, event_id_lh], (label_feed.shape[0], 1))
    #print 'event_series.shape:', event_series.shape
    events_info = mne.create_info([event_name_lh], params.FREQ_S, ch_types='eeg', montage=None)
    events_raw_mne = mne.io.RawArray(event_series_lh.T, info=events_info, verbose=None)
    events_lh_mne = mne.find_events(events_raw_mne, stim_channel=event_name_lh, verbose=None)


    # Create spatial filters
    picks = mne.pick_types(X_mne.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    #print 'picks:', picks
    X_epochs_rh = mne.Epochs(X_mne, events_rh_mne, event_id={'move': 1}, tmin=t_min, tmax=t_max, picks=picks, proj=False,
                          baseline=None, preload=True, add_eeg_ref=False, verbose=False)
    X_epochs_lh = mne.Epochs(X_mne, events_lh_mne, event_id={'move': 1}, tmin=t_min, tmax=t_max, picks=picks, proj=False,
                          baseline=None, preload=True, add_eeg_ref=False, verbose=False)
    X_epochs_concat = mne.epochs.concatenate_epochs([X_epochs_rh, X_epochs_lh])
    print 'X_epochs_rh._data.shape:', X_epochs_rh._data.shape
    print 'X_epochs_lh._data.shape:', X_epochs_lh._data.shape
    epoch_class_arr_rh = 1 * np.ones((X_epochs_rh._data.shape[0]), dtype=int)
    epoch_class_arr_lh = -1 * np.ones((X_epochs_rh._data.shape[0]), dtype=int)
    epoch_class_arr_concat = np.concatenate((epoch_class_arr_rh, epoch_class_arr_lh), axis=0)
    print 'epoch_class_arr_concat', epoch_class_arr_concat
    #X_epochs_concat = np.concatenate((X_epochs_rh, X_epochs_lh), axis=0)
    print 'X_epochs_rh_concat._data.shape:', X_epochs_concat._data.shape
    csp_obj = mne.decoding.CSP(n_components=4, reg='lws')
    csp_obj.fit(X_epochs_concat.get_data(), epoch_class_arr_concat)
    sp_filters = csp_obj.filters_[0:4].T
    print 'sp_filters.shape:', sp_filters.shape


    # Filter in spatial domain
    X_spfilt = np.dot(X_tdfilt, sp_filters)


    # Get the signal power
    X_pow_spf = X_spfilt ** 2


    # Get moving average on signal power
    X_ma_spf = signal.convolve(X_pow_spf.T, np.ones((1, int(1*params.FREQ_S)))).T


    # Plot the signal
    if False:
        #channels_to_plot = [1, 3, 9]
        #plt.plot(X_tdfilt[:, channels_to_plot])
        plt.plot(X_ma_spf[:, 0])
        #plt.plot(X_diff_c1_c2)
        #plt.plot(X_diff_c3_c4)
        #plt.plot(X_diff_c5_c6)
        plt.plot(100000*label_feed[:, 0:2])
        plt.xlim([1500, 5000])
        plt.show()


    # Plot topo
    if False:
        layout = mne.channels.read_layout('EEG1005')
        #X_epochs_avg = X_epochs_rh.average()
        #X_epochs_avg.plot_topomap(times=[0], ch_type='eeg', layout=layout)
        #mne.viz.plot_topomap(csp_obj.patterns_[0, :], X_epochs_concat)
        #X_epochs_avg = X_epochs.average()
        #X_epochs_avg.data = csp.patterns_.T
        #X_epochs_avg.times = np.arange(evoked.data.shape[0])
        #X_epochs_avg.plot_topomap(times=[0, 1, 2, 3, 4, 5], ch_type='eeg', layout=layout)
                                  #scale_time=1, time_format='%i', scale=1, unit='Patterns (AU)', size=1.5)
