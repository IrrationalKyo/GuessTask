'''
    This python script creates a model to infer number of tasks from busy intervals.
    Data Representation:
        Approach 1:
            Dataset creation procedure: Convert the trace into (timesteps, state) where there are three states (rest, busy,
            swich) for each time slot. State is a one-hot vector
        Approach 2:
            Dataset creation procedure: Convert the trace into (interval_slot, state) where three states (rest, busy, switch)
            have real value similar to one-hot vector (e.g. only one of the state can be non-zero)

    Model:
        Approach 1:
            LSTM + CNN in that order. The idea is for the LSTM to encode the data so that it preserves the temportal features
            and the CNN to exploit the local features
        Approach 2:
            Stacked DNN. This is for comparing the performance.
'''

class Converter:

    # returns 0 is state did not change
    # returns >0 if the state is still busy but switched
    # returns <0 if the state changed from busy to rest or rest to busy
    @staticmethod
    def state_changed(int_current, int_prev):
        if int_current - int_prev == 0:
            return 0
        elif int_current > 0 and int_prev > 0:
            return 1
        else:
            return -1

    # returns a list of intervals. (e.g. [(state, duration), (state, duration), (state, duration)])
    @staticmethod
    def vectorize(string_trace):
        # convert trace to intervals
        # state 0 = rest, 1 = busy, 2 = context switch
        vect = []
        int_interval_duration = 0
        int_prev_task = -1
        for i in range(len(string_trace)):
            int_interval_duration += 1
            int_current_task = string_trace[i]
            int_change_flag = Converter.state_changed(int_current_task, int_prev_task)
            if int_change_flag < 0:
                vect.append((0, int_interval_duration))
                int_interval_duration = 0
            elif int_change_flag > 0:
                # write interval
                vect.append((1, int_interval_duration))
                # write the context switch change.
                vect.append((2, 1))
                int_interval_duration = 0
            int_prev_task = int_current_task
        return vect

    # returns a randomly shifted convolution of data
    # i.e. a window which will be a data point will be randomly placed in the list of intervals
    # this will be used to multiply the dataset.
    # outputs
    @staticmethod
    def data_convolutor(vec_x, int_window_length):

        return

    # returns datapoint and its label (each data point is a list of vectorize() output from different dataset)
    @staticmethod
    def vectorize_data(vec_trace, float_train_ratio):
        #
        
        return
