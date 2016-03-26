"""



Links:
http://bastibe.de/2013-05-30-speeding-up-matplotlib.html

"""

from bme_bci_Recorder import Recorder
import mi_params
import threading
import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################

TAG = '[eeg_adjust] '
KEY_TERMINATE = 'q'
is_terminate_requested = False
PLT_PAUSE_TIME = 0.01




########################################################################################################################

def on_keypress(key_event):

    print 'key_event.key:', key_event.key
    if key_event.key == KEY_TERMINATE:
        is_terminate_requested = True




########################################################################################################################

class EegAdjust:

    def __init__(self):

        # Init data buffers
        self.len_buf = int(mi_params.FREQ_S)
        self.X_buf = np.zeros((self.len_buf, mi_params.NUM_CHANNELS))

        # Init subplots
        self.axis_x = np.arange(self.len_buf)
        self.num_plot_rows = 4
        self.num_plot_cols = mi_params.NUM_CHANNELS / self.num_plot_rows
        plt.ion()
        self.fig, self.axis_arr = plt.subplots(self.num_plot_rows, self.num_plot_cols, sharex=True, figsize=(800, 600))
        self.fig.canvas.mpl_connect('key_press_event', on_keypress)
        self.line_list = []
        for i_ch in range(mi_params.NUM_CHANNELS):
            line_temp, = self.axis_arr[i_ch % self.num_plot_rows, int(i_ch/self.num_plot_rows)].plot(self.axis_x, self.X_buf[:, i_ch])
            self.line_list.append(line_temp)
            self.axis_arr[i_ch % self.num_plot_rows, int(i_ch/self.num_plot_rows)].set_ylim([-50, 50])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(PLT_PAUSE_TIME)
        #print 'line_list len:', len(self.line_list)



    # End of init

    def plot_live(self):
        for i_ch in range(mi_params.NUM_CHANNELS):
            #self.axis_arr[i_ch % self.num_plot_rows, int(i_ch/self.num_plot_rows)].cla() slow
            #self.axis_arr[i_ch % self.num_plot_rows, int(i_ch/self.num_plot_rows)].plot(self.axis_x, self.X_buf[:, i_ch]) slow
            self.line_list[i_ch].set_ydata(self.X_buf[:, i_ch])
            axis_temp = self.axis_arr[i_ch % self.num_plot_rows, int(i_ch/self.num_plot_rows)]
            #axis_temp.draw_artist(axis_temp.patch)
            axis_temp.draw_artist(self.line_list[i_ch])
        self.fig.canvas.update()
        #self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(PLT_PAUSE_TIME)
        print TAG, 'Plot refreshed.'

    def start(self):
        # Init the amp
        print 'Initializing the amp...'
        recorder = Recorder('lslamp', mi_params.FREQ_S, mi_params.LEN_REC_BUF_SEC, mi_params.NUM_CHANNELS)
        thread_rec = threading.Thread(target=recorder.record)
        thread_rec.start()

        # Start plotting
        plt.show()
        #thread_plot = threading.Thread(target=self.plot_live)
        #thread_plot.start()

        # Get and display data from the amp
        counter = 0
        while not is_terminate_requested:
            data_last_chunk = recorder.get_new_data(mi_params.LEN_DATA_CHUNK_READ, mi_params.AMP_WAIT_SEC)
            recorder.acknowledge_new_data()
            print 'recorder.new_data_counter:', recorder.new_data_counter
            #print 'data_last_chunk.shape: ', data_last_chunk.shape
            i_row_from = int((counter*mi_params.LEN_DATA_CHUNK_READ) % self.len_buf)
            i_row_to = int(((counter+1)*mi_params.LEN_DATA_CHUNK_READ) % self.len_buf)
            if i_row_to == 0:
                i_row_to = self.len_buf
            print 'i_row_from, i_row_to:', i_row_from, i_row_to
            self.X_buf[i_row_from: i_row_to] = data_last_chunk
            #print 'X_buf[i_row_from: i_row_to]:\n', self.X_buf[i_row_from: i_row_to]

            self.plot_live()

            counter += 1

        # End of while

        # Stop the amp
        recorder.stop_recording()



########################################################################################################################


if __name__ == '__main__':

    print TAG, 'started.'


    eeg_adjust_obj = EegAdjust()
    eeg_adjust_obj.start()





# plt.ion() ## Note this correction
# fig=plt.figure()
# plt.axis([0,1000,0,1])

# i=0
# x=list()
# y=list()

# while i <1000:
    # temp_y=np.random.random();
    # x.append(i);
    # y.append(temp_y);
    # plt.scatter(i,temp_y);
    # i+=1;
    # plt.show()
    # plt.pause(0.0001) #Note this correction



    print TAG, 'terminates.'