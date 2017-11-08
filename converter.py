#from keras.datasets import imdb
import numpy as np
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
	converts the task list to numpy of shape (data_size, feature_size)
	
	label_card := cardinality of the label set
	label_size := number of slot predictions
'''
def list_to_array(trace_list, training_ratio, time_steps, label_card, label_size=1):
	# data_size := total number of data points that can be generated
	data_size = len(trace_list) - (time_steps + label_size)
	# training_size := number of training samples
	training_size = int(data_size * training_ratio)
	# test_size := number of test samples
	test_size = data_size - training_size

	# training_examples = np.zeros((training_size, time_steps, label_card))
	# training_labels = np.zeros((training_size, time_steps, label_card))

	# test_examples = np.zeros((test_size, time_steps, label_card))
	# test_labels = np.zeros((test_size, time_steps, label_card))

	training_examples, training_labels = sample_creater(trace_list[:training_size + time_steps +label_size],
														training_size,
														time_steps,
														label_card)
	test_examples, test_labels = sample_creater(trace_list[training_size:],
												  test_size,
												  time_steps,
												  label_card)

	return (training_examples, training_labels), (test_examples, test_labels)


'''
	Creates samples so each new example advances by 1 timestep (1 new label example)
'''
def sample_creater(trace_list, count, time_steps, label_card, label_size = 1):
	if count + time_steps + label_size > len(trace_list):
		raise ValueError("trace_list is too short."
						 "\tcount = " + str(count) +
						 "\tlabel_size =" + str(label_size) +
						 "\tlen(trace_list)" + str(len(trace_list)))
	if time_steps >= len(trace_list):
		raise ValueError("time_steps is larger than trace_list.\ntrace_list is too short"
			  "\ttime_steps = " + str(time_steps) +
			  "\tlen(trace_list)" + str(len(trace_list)))

	example = np.zeros((count, time_steps, label_card))
	label = np.zeros((count, time_steps, label_card))

	for l in range(count):
		for j in range(time_steps):
			''' TODO: find the correct index'''
			task_num = trace_list[l + j]
			example[l][j][task_num] = 1
			for k in range(1, label_size + 1):
				''' TODO: find the correct index'''
				task_num = trace_list[l + j + k]
				label[l][j][task_num] = 1
	return example, label

'''
	requires the length of the trace_list to be divisible by count.
'''
def sample_creater_disjoint(trace_list, count, time_steps, label_card, label_size = 1):

	example_len = (label_size + time_steps)

	example = np.zeros((count, time_steps, label_card))
	label = np.zeros((count, time_steps, label_card))

	for i in range(count):
		for j in range(time_steps):
			task_num = trace_list[i*(example_len) + j]
			example[i][j][task_num] = 1
			for k in range(1,1+label_size):
				label_num = trace_list[i*(example_len) + j + k]
				label[i][j][label_num] = 1

	return example, label


(train_x2, train_y2), (test_x2,test_y2) = list_to_array(text_to_list('dataset_1.txt'), 0.8, 100, 16)