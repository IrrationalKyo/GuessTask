#from keras.datasets import imdb
import numpy as np
import math
from multiprocessing import pool
import copy

# assume 2d array
# def printNumpy(array):


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
    input is of shape (time_steps, label_card)
    output is of shape (time_steps, label_card)
    Visual Representation (X = left column, Y = right column)
    t -           l tstepsOffset+
    s  -          c  tstepsOffset+
    t   -         a   tstepsOffset+
    e    -        r    tstepsOffset+
    p     -       d     tstepsOffset+
    s      -
'''
def list_to_example(trace_list, label_card, time_steps, label_size=1, offset=0):
    list_of_examples = []
    list_of_labels = []
    for i in range(len(trace_list)):
        if i + time_steps+offset+label_size >= len(trace_list):
            break

        # squash the mutiple steps
        example_datum = np.zeros((time_steps,label_card))
        for j in range(time_steps):
            task_num = trace_list[j+i]
            example_datum[j][int(task_num)] = 1
        list_of_examples.append(example_datum)

        label_datum = np.zeros((time_steps, label_card))
        for j in range(label_size):
            task_num = trace_list[i + time_steps + offset + j]
            label_datum[j][int(task_num)] = 1

        list_of_labels.append(label_datum)

    return (list_of_examples, list_of_labels)

'''
    This function requires the time
    overlap_gap: is the difference in the time index of the first slot between two data sequences
    Visual Representation (X = left column, Y = right column)
    t -           t gapOffset+
    s  -          s  gapOffset+
    t   -         t   gapOffset+
    e    -        e    gapOffset+
    p     -       p     gapOffset+
    s      -      s      gapOffset+
    
    keeping the gap==offset makes X[1] == Y[0]
'''
def list_to_example_overlap(trace_list, time_steps=100, offset=0, overlap_gap=1):

    label_card = detect_label_card(trace_list)
    list_of_examples = []
    list_of_labels = []

    #  TODO: PARALLZELIZE THIS, OR MAY BE NOT. THE PROBLEM WAS DUE TO LACK OF MEM
    for i in range(len(trace_list)):
        if i + time_steps+overlap_gap+offset >= len(trace_list):
            break

        # squash the mutiple steps
        example_datum = np.zeros((time_steps, label_card))
        for j in range(time_steps):
            task_num = trace_list[j + i]
            example_datum[j][int(task_num)] = 1
        list_of_examples.append(example_datum)

        label_datum = np.zeros((time_steps, label_card))
        for j in range(time_steps):
            task_num = trace_list[i + offset+ overlap_gap + j]
            label_datum[j][int(task_num)] = 1

        list_of_labels.append(label_datum)

    return (list_of_examples, list_of_labels)


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

def list_to_example_sequence(trace_list, offset=0, pred_len=1):
    label_card = detect_label_card(trace_list)
    list_of_examples = []
    list_of_labels = []
    sequence_len = len(trace_list)


    #  TODO: PARALLZELIZE THIS, OR MAY BE NOT. THE PROBLEM WAS DUE TO LACK OF MEM
    for i in range(sequence_len):
        if i + sequence_len + pred_len + offset >= len(trace_list):
            break

        example_datum = np.zeros(label_card)
        example_datum[int(trace_list[i])] = 1
        list_of_examples.append(example_datum)

        label_datum = np.zeros(label_card * pred_len)

        for j in range(pred_len):
            task_num = trace_list[i + offset + j + 1]
            label_datum[j * label_card + int(task_num)] = 1

        list_of_labels.append(label_datum)

    return (list_of_examples, list_of_labels)


def chunk_examples(examples, labels, start_index, end_index):
    if len(examples) != len(labels):
        raise ValueError("length of examples and labels do not match")

    chunk_size = end_index-start_index
    if chunk_size < 1:
        raise ValueError("chrunk_size has value of " + str(chunk_size))

    example_chunk = examples[start_index:end_index]
    label_fold = labels[start_index:end_index]

    label_dataset= np.asarray(label_fold, dtype=np.float32)
    example_dataset = np.asarray(example_chunk, dtype=np.float32)

    return (example_dataset, label_dataset)


def generate_folds(folds, data_pair):
    example_list = data_pair[0]
    label_list = data_pair[1]

    if len(example_list) != len(label_list):
        raise ValueError("expected example and label to have same number of samples")

    fold_size = math.floor(len(example_list)/folds)

    result = []
    for i in range(folds):
        fold = chunk_examples(example_list, label_list, i*fold_size, (i+1)*fold_size)
        result.append(fold)

    return result

'''
    This is the correct way
'''
def generate_time_series_folds(folds, data_pair, batch_size=1):
    example_list = data_pair[0]
    label_list = data_pair[1]
    partition = folds + 1
    if len(example_list) != len(label_list):
        raise ValueError("expected example and label to have same number of samples")

    partition_size = math.floor(len(example_list) / partition)
    while partition_size % batch_size != 0:
        partition_size -= 1

    ''' list of lists. each element of result is a list of partition where the last parition is the testing set'''
    result = []

    for i in range(folds):
        fold = []
        fold.append(chunk_examples(example_list, label_list, 0, (i+1)*partition_size))
        fold.append(chunk_examples(example_list, label_list, (i + 1) * partition_size , (i + 2) * partition_size))
        result.append(fold)

    return result

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


if __name__ == "__main__":
    # res = generate_folds2(5,list_to_example(text_to_list('dataset_1.txt'),16,100,2))
    # print(validate_dataset(res[0][0],res[0][1],0))
    # res = generate_time_series_folds(5, list_to_example_overlap_100(text_to_list('dataset_det_1.txt'),16), batch_size = 10)
    # for fold in res:
    # 	print("x shape: " + str(fold[0][0].shape))
    # 	print("y shape: " + str(fold[0][1].shape))
    # 	print("end")

    # arr = np.zeros(16)
    #
    # for i in text_to_list('dataset_new_det_1.txt'):
    # 	arr[i] += 1
    # print(arr)
    # sortedarr = np.argsort(arr)
    # print("least frequent to most frequent")
    # for i in range(16):
    # 	print(str(sortedarr[i]))

    np.set_printoptions(threshold=np.inf)

    # example = list_to_example(newText_to_list("data/dataset_det_1.txt")[0:100],16,time_steps=10, label_size=1, offset=0)
    example = list_to_example_overlap(newText_to_list("data/size29rep4.data"), time_steps=10, offset=0, overlap_gap=10)

    dataset1 = chunk_examples(example[0], example[1], 0, 10000)
    dataset2 = chunk_examples(example[0], example[1], 0, 10000)

    if np.array_equal(dataset1[0],dataset1[0]):
        print("EQUAL")
    else:
        print("NOT_EQUAL")