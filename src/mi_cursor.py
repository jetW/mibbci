'''

TODO:
ratio, not difference
deadband instead of rest state calib
analyze signal during color crossfade as well
more intensive movements
try different limbs
try a sportsman who is used to limb control, sportlovo

'''
from bme_bci_Recorder import Recorder
import mi_params
import graphics
import math
import threading
import numpy as np
import time
from datetime import datetime
#import csv
from scipy import signal
#from joblib import Parallel, delayed
import matplotlib.pyplot as plt




########################################################################################################################

# 1-16 INDEXING, NOT 0-15: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP5, CPz, CP6, P3, Pz, P4

#CONFIG_FILE_NAME = 'config_test.txt'
is_simulation_mode = False
if is_simulation_mode:
    FEAT_MULT_1 = mi_params.FEAT_MULT_1_SIMU
    FEAT_MULT_2 = mi_params.FEAT_MULT_2_SIMU
else:
    FEAT_MULT_1 = mi_params.FEAT_MULT_1_REAL
    FEAT_MULT_2 = mi_params.FEAT_MULT_2_REAL





########################################################################################################################

def cursor_func():

    print 'cursor_func(.) entered.'
    cursor_radius = 26
    w = 2 * math.pi / 10


    # Initialize the time-domain filter
    freq_Nyq = mi_params.FREQ_S/2.
    freq_trans = 0.5
    freqs_FIR_Hz = np.array([8.-freq_trans, 12.+freq_trans])
    #numer = signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = signal.firwin(mi_params.M_FIR, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = 1.
    '''w, h = signal.freqz(numer)
    plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()'''


    # Set up graphics
    win = graphics.GraphWin('Cursor', mi_params.IMAGE_W, mi_params.IMAGE_H)
    cursor = graphics.Circle(graphics.Point(mi_params.IMAGE_W/2, mi_params.IMAGE_H/2), cursor_radius)
    #cursor.setFill(CURSOR_COLOR_REST)
    #cursor.setOutline(CURSOR_COLOR_REST)
    cursor.setFill(graphics.color_rgb(mi_params.CURSOR_COLOR_REST[0], mi_params.CURSOR_COLOR_REST[1], mi_params.CURSOR_COLOR_REST[2]))
    cursor.setOutline(graphics.color_rgb(mi_params.CURSOR_COLOR_REST[0], mi_params.CURSOR_COLOR_REST[1], mi_params.CURSOR_COLOR_REST[2]))
    cursor.draw(win)
    cursor_pos_prev = np.array([mi_params.IMAGE_W/2, mi_params.IMAGE_H/2])
    cursor_pos = cursor_pos_prev
    #my_canvas.delete('all')
    #event_arr = np.zeros(((LEN_IDLE_SEC+LEN_RIGHT_SEC+LEN_IDLE_SEC+LEN_LEFT_SEC), 3))
    event_arr_right = np.zeros((mi_params.LEN_DATA_CHUNK, mi_params.NUM_EVENT_TYPES));
    event_arr_right[:, 0] = np.ones(mi_params.LEN_DATA_CHUNK);        # TODO event ids
    event_arr_left = np.zeros((mi_params.LEN_DATA_CHUNK, mi_params.NUM_EVENT_TYPES));
    event_arr_left[:, 1] = np.ones(mi_params.LEN_DATA_CHUNK);        # TODO event ids
    event_arr_idle = np.zeros((mi_params.LEN_DATA_CHUNK, mi_params.NUM_EVENT_TYPES));
    event_arr_idle[:, 2] = np.ones(mi_params.LEN_DATA_CHUNK);        # TODO event ids
    event_arr_calib = np.zeros((mi_params.LEN_DATA_CHUNK, mi_params.NUM_EVENT_TYPES));
    event_arr_calib[:, 3] = np.ones(mi_params.LEN_DATA_CHUNK);        # TODO event ids
    #cursor_color_list = []
    cursor_event_list = []
    cursor_color_arr_raw = np.zeros((int(mi_params.LEN_PERIOD_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK), 3))
    color_counter = 0
    for i in range(int(mi_params.LEN_IDLE_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)):
        #cursor_color_list.append(CURSOR_COLOR_IDLE)
        cursor_color_arr_raw[color_counter, :] = mi_params.CURSOR_COLOR_IDLE
        cursor_event_list.append(event_arr_idle)      # r, l, idle, calib
        color_counter += 1
    for i in range(int(mi_params.LEN_RIGHT_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)):
        #cursor_color_list.append(CURSOR_COLOR_RIGHT)
        cursor_color_arr_raw[color_counter, :] = mi_params.CURSOR_COLOR_RIGHT
        cursor_event_list.append(event_arr_right)
        color_counter += 1
    for i in range(int(mi_params.LEN_IDLE_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)):
        #cursor_color_list.append(CURSOR_COLOR_IDLE)
        cursor_color_arr_raw[color_counter, :] = mi_params.CURSOR_COLOR_IDLE
        cursor_event_list.append(event_arr_idle)
        color_counter += 1
    for i in range(int(mi_params.LEN_LEFT_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)):
        #cursor_color_list.append(CURSOR_COLOR_LEFT)
        cursor_color_arr_raw[color_counter, :] = mi_params.CURSOR_COLOR_LEFT
        cursor_event_list.append(event_arr_left)
        color_counter += 1
    conv_window = np.ones((mi_params.LEN_COLOR_CONV_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK, 1))\
                  / (int(mi_params.LEN_COLOR_CONV_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK))
    cursor_color_arr_ud = np.flipud(cursor_color_arr_raw)
    cursor_color_arr_ud_convd = signal.convolve(cursor_color_arr_ud.T, conv_window.T).T
    cursor_color_arr_final = np.flipud(cursor_color_arr_ud_convd[0:cursor_color_arr_raw.shape[0], :])
    if False:
        plt.figure()
        plt.plot(cursor_color_arr_raw)
        #plt.plot(cursor_color_arr_ud[:, 0])
        #plt.plot(cursor_color_arr_ud_convd[:, 0])
        plt.plot(cursor_color_arr_final)
        #plt.legend(['raw', 'ud', 'ud_convd', 'final'])
        plt.show()


    # Initialize the amplifier
    if not is_simulation_mode:
        print 'Initializing the amp...'
        recorder = Recorder('lslamp', mi_params.FREQ_S, mi_params.LEN_REC_BUF_SEC, mi_params.NUM_CHANNELS)
        thread_rec = threading.Thread(target=recorder.record)
        thread_rec.start()


    # Wait a little until the recorder gets some data
    #time.sleep(2.0)
    if not is_simulation_mode:
        while recorder.new_data_counter < 20:
            print 'Waiting for some initial data...'
            time.sleep(0.1)


    # Rest-state offset estimation loop
    #X_buf_rest_est = np.zeros((1.2*REST_EST_LEN_SAMPLES, 2))    # 2 relevant channels
    X_buf_raw_live = np.zeros((mi_params.FREQ_S*mi_params.LEN_REC_BUF_SEC, mi_params.NUM_CHANNELS))
    X_events_live = np.zeros((mi_params.FREQ_S*mi_params.LEN_REC_BUF_SEC, mi_params.NUM_EVENT_TYPES))
    X_buf_live = np.zeros((mi_params.FREQ_S*mi_params.LEN_REC_BUF_SEC, 2))
    X_buf_feat_live = np.zeros((mi_params.FREQ_S*mi_params.LEN_REC_BUF_SEC, 1))
    X_feat_log = np.zeros((mi_params.FREQ_S*mi_params.LEN_REC_BUF_SEC/mi_params.LEN_DATA_CHUNK, 1))
    counter = 0
    while counter < (mi_params.LEN_REST_ESTIM_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK):
        print 'counter: ', counter

        # Clear the canvas
        win.delete('all')

        #if recorder.is_new_data_available:

        if not is_simulation_mode:
            # Wait for new data and get it
            data_last_chunk = recorder.get_new_data(mi_params.LEN_DATA_CHUNK, 0.01)
            recorder.acknowledge_new_data()
            print 'recorder.new_data_counter:', recorder.new_data_counter
        else:
            time.sleep(1.0 / (mi_params.FREQ_S/mi_params.LEN_DATA_CHUNK))
            data_last_chunk = 1000.0 * np.random.rand(mi_params.LEN_DATA_CHUNK, mi_params.NUM_CHANNELS)
            #print 'Random data_last_chunk size:', data_last_chunk

        if True:
            # Insert the new sample to our time series
            X_buf_raw_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :]\
                            = data_last_chunk
            #print 'data_last_chunk:', data_last_chunk
            #data_last_chunk = np.random.rand(LEN_DATA_CHUNK, NUM_CHANNELS)
            #print 'data_last_chunk.shape:', data_last_chunk.shape
            #print '(data_last_chunk[:, 7]-data_last_chunk[:, 3]).shape:', (data_last_chunk[:, 7]-data_last_chunk[:, 3]).shape
            #data_last_reref = data_last_chunk[:, (7, 10)]
            #print 'data_last_reref.shape:', data_last_reref.shape
            data_last_reref = np.array([ (data_last_chunk[:, 7]-data_last_chunk[:, 4]), (data_last_chunk[:, 10]-data_last_chunk[:, 4])]).T  # Re-reference
            #print 'data_last_reref.shape:', data_last_reref.shape
            #print 'data_last_reref:', data_last_reref
            X_buf_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :] = data_last_reref   # TODO which channel ids
            #print 'X_buf_live[((counter+LEN_PADDING)*LEN_DATA_CHUNK):((counter+LEN_PADDING+1)*LEN_DATA_CHUNK), :]:', X_buf_live[((counter+LEN_PADDING)*LEN_DATA_CHUNK):((counter+LEN_PADDING+1)*LEN_DATA_CHUNK), :]
            X_events_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :] = event_arr_calib

            # Process the data
            i_2 = (counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK
            #print 'numer, denom:', numer, denom
            #print 'X_buf_live[(i_2-M_FIR):i_2, :]', X_buf_live[(i_2-M_FIR):i_2, :]
            X_filt = signal.lfilter(numer, denom, X_buf_live[(i_2-mi_params.M_FIR):i_2, :].T).T
            #X_to_filt = X_buf_live[(i_2-M_FIR):i_2, :]
            #print 'X_to_filt.shape:', X_to_filt.shape
            #X_filt = np.array(Parallel(n_jobs=4)(delayed(signal.lfilter)
            #        (numer, denom, X_to_filt[:, ch]) for ch in range(X_to_filt.shape[1]))).T
            #print 'X_filt:', X_filt
            #print 'X_filt.shape:', X_filt.shape
            X_pow = X_filt ** 2
            #print 'X_pow:', X_pow
            X_pow_mean = np.mean(X_pow, axis=0)
            #print 'X_pow mean:', X_pow_mean
            X_pow_diff = X_pow_mean[0] - X_pow_mean[1]

            #cursor_pos = graphics.Point(IMAGE_W/2 + 100*math.cos(w*counter), IMAGE_H/2)
            #diff_mult = 4000.0  # Ok for simulation
            X_feat = FEAT_MULT_1 * X_pow_diff
            print 'X_feat (rest): ', X_feat
            X_feat_log[counter] = X_feat
            X_buf_feat_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :]\
                            = X_feat * np.ones((mi_params.LEN_DATA_CHUNK, 1))
            if counter > mi_params.IMP_RESP_LEN:
                cursor_pos = cursor_pos_prev + np.array([X_feat, 0])
                print 'cursor_pos: ', cursor_pos

            cursor_pos_point = graphics.Point(cursor_pos[0], cursor_pos[1])
            cursor_pos_prev = cursor_pos
            cursor = graphics.Circle(cursor_pos_point, cursor_radius)
            #cursor.setFill(CURSOR_COLOR_REST)
            #cursor.setOutline(CURSOR_COLOR_REST)
            cursor.setFill(graphics.color_rgb(mi_params.CURSOR_COLOR_REST[0], mi_params.CURSOR_COLOR_REST[1], mi_params.CURSOR_COLOR_REST[2]))
            cursor.setOutline(graphics.color_rgb(mi_params.CURSOR_COLOR_REST[0], mi_params.CURSOR_COLOR_REST[1], mi_params.CURSOR_COLOR_REST[2]))
            cursor.draw(win)

            #time.sleep(1.0 / (FREQ_S/LEN_DATA_CHUNK)) no
            #win.getMouse()
            counter += 1

        # End of if
    # End of while


    # Get the average rest state offset
    X_feat_rest_offset = np.mean(X_feat_log[(counter-(mi_params.FREQ_S*5.0/mi_params.LEN_DATA_CHUNK)):counter])
    print 'X_feat_log[0:counter]: ', X_feat_log[0:counter]
    print 'X_feat_rest_offset: ', X_feat_rest_offset


    # Cursor control loop
    #counter = 0 go on
    #while True:
    while counter < (mi_params.LEN_REC_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK):
        print 'counter: ', counter

        # Clear the canvas
        win.delete('all')

        if not is_simulation_mode:
            # Wait for new data and get it
            data_last_chunk = recorder.get_new_data(mi_params.LEN_DATA_CHUNK, 0.01)
            recorder.acknowledge_new_data()
            print 'recorder.new_data_counter:', recorder.new_data_counter
        else:
            time.sleep(1.0 / (mi_params.FREQ_S/mi_params.LEN_DATA_CHUNK))
            data_last_chunk = 1000.0 * np.random.rand(mi_params.LEN_DATA_CHUNK, mi_params.NUM_CHANNELS)
            #print 'Random data_last_chunk size:', data_last_chunk

        if True:
            # Insert the new sample to our time series
            X_buf_raw_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :] = data_last_chunk
            #print 'data_last_chunk:', data_last_chunk
            #data_last_chunk = np.random.rand(LEN_DATA_CHUNK, NUM_CHANNELS)
            #print 'data_last_chunk.shape:', data_last_chunk.shape
            #print '(data_last_chunk[:, 7]-data_last_chunk[:, 3]).shape:', (data_last_chunk[:, 7]-data_last_chunk[:, 3]).shape
            #data_last_reref = data_last_chunk[:, (7, 10)]
            #print 'data_last_reref.shape:', data_last_reref.shape
            data_last_reref = np.array([ (data_last_chunk[:, 7]-data_last_chunk[:, 4]), (data_last_chunk[:, 10]-data_last_chunk[:, 4])]).T  # Re-reference
            #print 'data_last_reref.shape:', data_last_reref.shape
            #print 'data_last_reref:', data_last_reref
            X_buf_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :]\
                            = data_last_reref   # TODO which channel ids
            #print 'X_buf_live[((counter+LEN_PADDING)*LEN_DATA_CHUNK):((counter+LEN_PADDING+1)*LEN_DATA_CHUNK), :]:', X_buf_live[((counter+LEN_PADDING)*LEN_DATA_CHUNK):((counter+LEN_PADDING+1)*LEN_DATA_CHUNK), :]
            X_events_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :]\
                            = cursor_event_list[counter % int(mi_params.LEN_PERIOD_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)]

            # Process the data
            i_2 = (counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK
            #print 'numer, denom:', numer, denom
            #print 'X_buf_live[(i_2-M_FIR):i_2, :]', X_buf_live[(i_2-M_FIR):i_2, :]
            X_filt = signal.lfilter(numer, denom, X_buf_live[(i_2-mi_params.M_FIR):i_2, :].T).T
            #X_to_filt = X_buf_live[(i_2-M_FIR):i_2, :]
            #print 'X_to_filt.shape:', X_to_filt.shape
            #X_filt = np.array(Parallel(n_jobs=4)(delayed(signal.lfilter)
            #        (numer, denom, X_to_filt[:, ch]) for ch in range(X_to_filt.shape[1]))).T
            #print 'X_filt:', X_filt
            #print 'X_filt.shape:', X_filt.shape
            X_pow = X_filt ** 2
            #print 'X_pow:', X_pow
            X_pow_mean = np.mean(X_pow, axis=0)
            print 'X_pow mean:', X_pow_mean
            X_pow_diff = X_pow_mean[0] - X_pow_mean[1]

            #cursor_pos = graphics.Point(IMAGE_W/2 + 100*math.cos(w*counter), IMAGE_H/2)
            #diff_mult = 4000.0  # Ok for simulation
            X_feat_pre = FEAT_MULT_1*X_pow_diff - X_feat_rest_offset
            print 'X_feat_pre: ', X_feat_pre
            X_feat = FEAT_MULT_2 * X_feat_pre
            print 'X_feat: ', X_feat
            X_feat_log[counter] = X_feat
            X_buf_feat_live[((counter+mi_params.LEN_PADDING)*mi_params.LEN_DATA_CHUNK):((counter+mi_params.LEN_PADDING+1)*mi_params.LEN_DATA_CHUNK), :]\
                            = X_feat * np.ones((mi_params.LEN_DATA_CHUNK, 1))
            if counter > mi_params.IMP_RESP_LEN:
                cursor_pos = cursor_pos_prev + np.array([X_feat, 0])
                print 'cursor_pos: ', cursor_pos

            cursor_pos_point = graphics.Point(cursor_pos[0], cursor_pos[1])
            cursor_pos_prev = cursor_pos
            cursor = graphics.Circle(cursor_pos_point, cursor_radius)
            #cursor.setFill(cursor_color_list[counter % int(LEN_PERIOD_SEC * FREQ_S / LEN_DATA_CHUNK)])
            #cursor.setOutline(cursor_color_list[counter % int(LEN_PERIOD_SEC * FREQ_S / LEN_DATA_CHUNK)])
            color_temp = cursor_color_arr_final[counter % int(mi_params.LEN_PERIOD_SEC * mi_params.FREQ_S / mi_params.LEN_DATA_CHUNK)]
            cursor.setFill(graphics.color_rgb(color_temp[0], color_temp[1], color_temp[2]))
            cursor.setOutline(graphics.color_rgb(color_temp[0], color_temp[1], color_temp[2]))
            cursor.draw(win)

            #time.sleep(1.0 / (FREQ_S/LEN_DATA_CHUNK)) no
            #win.getMouse()
            counter += 1

        # End of if
    # End of while


    # Stop recording
    recorder.stop_recording()


    # Close the window
    win.close()




    # Save data to file
    time_axis = np.arange(X_buf_live.shape[0]).reshape((X_buf_live.shape[0], 1))
    print 'time_axis.shape:', time_axis.shape
    #rec_data, rec_times = recorder.get_data() no cuz of simu
    #data_merged = np.concatenate((time_axis, X, marker_axis_arr), axis=1)
    data_merged = np.concatenate((time_axis, X_buf_raw_live, X_events_live, X_buf_live, X_buf_feat_live), axis=1)
    print 'data_merged.shape: ', data_merged.shape
    time_save = datetime.now()
    np.savetxt('BME_BCI_MI_REC_{0}{1:02}{2:02}_{3:02}h{4:02}m{5:02}s.csv'.format(time_save.year, time_save.month, time_save.day,
               time_save.hour, time_save.minute, time_save.second),
               X=data_merged, fmt='%.8f', delimiter=",",
               header=str(1), comments='time, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, rh, lh, idle, calib, b1, b2, feat')  # TODO


    print 'cursor_func(.) terminates.'




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'


    # Start the threads
    #thread_rec = threading.Thread(target=recorder.record)
    #thread_cursor = threading.Thread(target=cursor_func)
    #thread_rec.start()
    #thread_cursor.start()
    cursor_func()




    print 'Main terminates.'
