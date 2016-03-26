

from bme_bci_Recorder import Recorder
import mi_params
import threading
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':

    # Init plot axes
    axis_x = np.arange(int(mi_params.FREQ_S) - 1)
    #axis_y = np.linspace(-50.0, 50.0, num=101)
    y_test = np.random.rand(axis_x.shape[0])

    # Two subplots, the axes array is 1-d
    fig, axis_arr = plt.subplots(4, mi_params.NUM_CHANNELS/4, sharex=True)
    axis_arr[0, 0].plot(axis_x, y_test)
    axis_arr[0, 0].set_title('Sharing X axis')
    axis_arr[1, 0].scatter(axis_x, y_test)
    plt.show()


    """# Init the amp
    print 'Initializing the amp...'
    recorder = Recorder('lslamp', mi_params.FREQ_S, mi_params.LEN_REC_BUF_SEC, mi_params.NUM_CHANNELS)
    thread_rec = threading.Thread(target=recorder.record)
    thread_rec.start()


    # Do


    # Stop the amp
    recorder.stop_recording()"""



"""plt.ion() ## Note this correction
fig=plt.figure()
plt.axis([0,1000,0,1])

i=0
x=list()
y=list()

while i <1000:
    temp_y=np.random.random();
    x.append(i);
    y.append(temp_y);
    plt.scatter(i,temp_y);
    i+=1;
    plt.show()
    plt.pause(0.0001) #Note this correction"""