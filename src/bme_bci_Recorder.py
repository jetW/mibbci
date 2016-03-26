
import multiprocessing
import libmushu
import numpy as np
import time
from datetime import datetime


#NUM_TIME_ELEMENTS = 7


class Recorder:

    def __init__(self, amp_name, freq_sampling, len_buf_s, num_channels):

        multiprocessing.freeze_support()    # Needed on Windows

        #available_amps = libmushu.get_available_amps()
        #print('available_amps:', available_amps)
        #amp_name = available_amps[0]
        #print 'amp_name:', amp_name
        #amp = libmushu.get_amp(amp_name)
        #cfg = amp.presets
        #print 'cfg:', cfg
        #amp.configure(cfg)
        #amp.configure(mode='data', fs=128, channels=16)
        #amp.configure() no


        # Initialize gUSBamp
        #amp = libmushu.get_amp('gusbamp') it does not find it like this
        self.amp = libmushu.get_amp(amp_name)
        #amp.configure(fs=128, mode='recording')
        self.amp.configure() # Sampling freq is set in the gUSBamp client program
        self.amp.start()
        self.freq_s = freq_sampling


        # Initialize numpy buffers
        self.rec_data = np.zeros((len_buf_s * freq_sampling, num_channels))
        self.rec_times = np.zeros(len_buf_s * freq_sampling)
        print 'self.rec_data.shape at init:', self.rec_data.shape


        # Initialize flags
        self.is_new_data_available = False
        self.is_rec_stopped = False
        self.is_rec_finished = False


        # Other
        self.i_time_fast = 0
        self.new_data_counter = 0


    # End of init

    def get_data(self):
        return self.rec_data, self.rec_times

    def get_new_data(self, last_len, wait_s):
        # Wait for new data
        while not self.is_new_data_available:
            time.sleep(wait_s)

        if self.rec_data.shape[0] >= last_len:
            #print 'self.rec_data.shape:', self.rec_data.shape
            return self.rec_data[(self.i_time_fast-last_len):self.i_time_fast, :]
        else:
            print 'Error: Requested length exceeds record length.'
            return -1

    def get_last_data(self, last_len):
        if self.rec_data.shape[0] >= last_len:
            #print 'self.rec_data.shape:', self.rec_data.shape
            return self.rec_data[(self.i_time_fast-last_len):self.i_time_fast, :]
        else:
            print 'Error: Requested length exceeds record length.'
            return -1

    def record(self):
        i_time_atomic = 0
        self.i_time_fast = 0

        #print 'self.is_rec_stopped before while...:', self.is_rec_stopped
        while not self.is_rec_stopped:
            #print 'i_time:', i_time

            time.sleep(1.0/self.freq_s)
            time_temp = datetime.now()
            data, marker = self.amp.get_data()
            #print data.shape, len(marker)
            #print type(data)
            #print '[Recorder] data.shape:', data.shape[0], data.shape[1]
            #print data[0:4]

            data_len = data.shape[0]
            if data_len > 0:
                #print 'Recorder new data.shape:', data.shape
                #print 'Recorder new data:', data
                self.rec_data[i_time_atomic:(i_time_atomic+data_len), :] = data
                self.i_time_fast += data_len
                self.is_new_data_available = True
                self.new_data_counter += 1

                #print 'time_temp:', time_temp, time_temp.hour, time_temp.minute, time_temp.second, time_temp.microsecond/1000
                time_flat = 3600000*time_temp.hour + 60000*time_temp.minute + 1000*time_temp.second + time_temp.microsecond/1000
                #print 'time_flat:', time_flat
                times_arr = (np.ones(data_len) * time_flat) + np.arange(-(data_len-1), 1)*(1000.0/self.freq_s)
                #print 'times_arr:', times_arr, 'i_time:', i_time
                self.rec_times[i_time_atomic:(i_time_atomic+data_len)] = times_arr

                i_time_atomic += data_len


        # Stop the recording
        self.amp.stop()
        print 'amp stopped.'


        # Resize the data buffer to the actual data size
        self.rec_data = self.rec_data[:i_time_atomic, :]
        self.rec_times = self.rec_times[:i_time_atomic]
        self.rec_times = self.rec_times.reshape((self.rec_times.shape[0], 1))
        print 'Final i_time:', i_time_atomic
        print 'Final self.rec_data.shape:', self.rec_data.shape
        print 'Final self.rec_times.shape:', self.rec_times.shape

        # Set the "finished" flag
        self.is_rec_finished = True
        print 'record(.) terminates.'

    def acknowledge_new_data(self):
        self.is_new_data_available = False
        self.new_data_counter -= 1

    def stop_recording(self):
        self.is_rec_stopped = True
        print 'Recording stopped, self.is_rec_stopped:', self.is_rec_stopped

    def is_finished(self):
        return self.is_rec_finished