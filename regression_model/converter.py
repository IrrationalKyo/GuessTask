import numpy as np
import random
import math


def text_to_list(fileName):
    trace = []
    data_file = open(fileName, 'r')
    raw_trace = data_file.readlines()[0].split(',')

    time_range = None
    for i in range(len(raw_trace)):
        if i%3 == 0:
            raw_time_range = raw_trace[i].split("-")
            time_range = (int(raw_time_range[0]),int(raw_time_range[1]))
        elif i%3 == 1:
            task_id = int(raw_trace[i])
            for i in range(time_range[1] - time_range[0]):
                trace.append(task_id)
        else:
            time_range = None
    return trace

'''
    This function assumes that labels are in a integer sequence from 0 to n. No skipping of a seuqence.
'''
def detect_label_card(trace):
    dic = {}
    for tr in trace:
        dic[str(tr)]=1
    return len(dic)

def newText_to_list(fileName):
    data_file = open(fileName, 'r')
    sequences = data_file.readlines()[0].split(',')
    trace = list(map(int, list(map(str.strip,sequences))))
    return trace

'''
    This function requires the time
    Visual Representation (X = left column, Y = right column)
    t -           t gapOffset+
    s  -          s  gapOffset+
    t   -         t   gapOffset+
    e    -        e    gapOffset+
    p     -       p     gapOffset+
    s      -      s      gapOffset+

    keeping the gap==offset makes X[1] == Y[0]
'''
def list_to_example_regression(trace_X, trace_Y, timesteps = 1, offset=0, pred_len=1, seed=0):
    old_length = len(trace_X)
    trace_list_X = cut_random(trace_X, int(old_length * 0.1), int(old_length * 0.01), seed)
    trace_list_Y = cut_random(trace_Y, int(old_length * 0.1), int(old_length * 0.01), seed)

    if len(trace_list_X) != len(trace_list_Y):
        print("THE LIST LENGTH IS DIFFERENT")
        return

    list_of_examples = []
    list_of_labels = []
    sequence_len = len(trace_list_X)


    for i in range(sequence_len):
        if i + pred_len + timesteps + offset + 1>= len(trace_list_X):
            break
        example_datum = np.zeros(timesteps)
        for j in range(timesteps):
            example_datum[j] = trace_list_X[i+j]
        list_of_examples.append(example_datum)

        label_datum = np.zeros(1)
        for j in range(1):
            prediction_value = trace_list_Y[i + offset + j]
            label_datum[j] = prediction_value

        list_of_labels.append(label_datum)

    if len(list_of_examples) != len(list_of_labels):
        print("THE LIST LENGTH IS DIFFERENT")
        return


    return (list_of_examples, list_of_labels)

def list_to_example_regression_ss(trace_X, offset=0, pred_len=1, seed=0):
    old_length = len(trace_X)
    trace_list_X = cut_random(trace_X, int(old_length * 0.1), int(old_length * 0.01), seed)

    list_of_examples = []
    list_of_labels = []
    sequence_len = len(trace_list_X)


    for i in range(sequence_len):
        if i + pred_len + offset + 1>= len(trace_list_X):
            break

        example_datum = np.zeros(1)
        example_datum[0] = trace_list_X[i]
        list_of_examples.append(example_datum)

        label_datum = np.zeros(pred_len)
        for j in range(1):
            prediction_value = trace_list_X[i + offset + j + 1]
            label_datum[j] = prediction_value

        list_of_labels.append(label_datum)

    return (list_of_examples, list_of_labels)

# cutoff first few sequences of trace
def cut_random(trace, upper_bound, lower_bound, seed=0):
    random.seed(seed)
    offset = random.randint(lower_bound, upper_bound)
    return trace[offset:]


def chunk_examples(examples, labels, start_index, end_index):
    print(len(examples))


    chunk_size = end_index-start_index
    if chunk_size < 1:
        raise ValueError("chrunk_size has value of " + str(chunk_size))

    example_chunk = examples[start_index:end_index]
    label_fold = labels[start_index:end_index]

    label_dataset= np.asarray(label_fold, dtype=np.float32)
    example_dataset = np.asarray(example_chunk, dtype=np.float32)

    return example_dataset, label_dataset

def index_of_label(vec):
    for i in range(len(vec)):
        if vec[i] == 1:
            return i
    return -1

''' currently only checks correctly when label_size = 1'''
def validate_dataset(example_dataset, label_dataset, offset=0):
    label_size = label_dataset[0].shape[0]

    for i in range(len(example_dataset)-(offset + label_size)):
        for j in range(label_size):
            index = index_of_label(label_dataset[i][j])
            if example_dataset[i+j+1+offset][-1][index] != 1:
                print("ERROR")
                print("label_vec: " + str(label_dataset[i][j]))
                print("example_vec: " + str(example_dataset[i+j+1+offset][0][index]))
                print("i: " + str(i) + " j: " + str(j) + " ")
                print("label should be " + str(index))
                print("training task is " + str(example_dataset[i+j+1+offset][0][index]))

                return -1
    return 0

def mySim_to_list(fileName):
    trace = []
    data_file = open(fileName, 'r')
    raw_trace = data_file.readlines()[0].split(',')


    for i in range(len(raw_trace)):
        trace.append(int(raw_trace[i]))
    return trace

def split_train_test(ratio, data_pair, timesteps=1):
    example_list = data_pair[0]
    label_list = data_pair[1]
    train_max_ind =  int(len(example_list) * ratio)

    print(len(example_list))
    print(len(label_list))
    if len(example_list) != len(label_list):
        raise ValueError("expected example and label to have same number of samples")

    train_x, train_y = chunk_examples(example_list, label_list, 0, train_max_ind)
    test_x, test_y = chunk_examples(example_list, label_list, train_max_ind+1, len(example_list)-1)

    return train_x, train_y, test_x, test_y


# if __name__ == "__main__":
    #
    # trace = newText_to_list("./data/size15rep0.data")
    # example = list_to_example_sequence(trace,offset=0, pred_len=1)
    #
    # trace_len = len(example[0])
    #
    # dataset1 = chunk_examples(example[0], example[1], 0, trace_len)
    #
    # print(split_train_test(0.75, dataset1))
    #
