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

		label_datum = np.zeros((label_size, label_card))
		for j in range(label_size):
			task_num = trace_list[i + time_steps + offset + j]
			label_datum[j][int(task_num)] = 1

		list_of_labels.append(label_datum)

	return (list_of_examples, list_of_labels)



'''
	converts the task list to numpy of shape (data_size, feature_size)
	
	label_card := cardinality of the label set
	label_size := number of slot predictions
	offset := number of slots ahead of the next slot
'''
# def list_to_array(trace_list, training_ratio, time_steps, label_card, label_size=1, offset = 0):
# 	# data_size := total number of data points that can be generated
# 	data_size = len(trace_list) - (time_steps + label_size)
# 	# training_size := number of training samples
# 	training_size = int(data_size * training_ratio)
# 	# test_size := number of test samples
# 	test_size = data_size - training_size
#
# 	# training_examples = np.zeros((training_size, time_steps, label_card))
# 	# training_labels = np.zeros((training_size, time_steps, label_card))
#
# 	# test_examples = np.zeros((test_size, time_steps, label_card))
# 	# test_labels = np.zeros((test_size, time_steps, label_card))
#
# 	training_examples, training_labels = sample_creater(trace_list[:training_size + time_steps +label_size],
# 														training_size,
# 														time_steps,
# 														label_card,
# 														offset=offset)
# 	test_examples, test_labels = sample_creater(trace_list[training_size:],
# 												test_size,
# 												time_steps,
# 												label_card,
# 												offset=offset)
#
# 	return (training_examples, training_labels), (test_examples, test_labels)


# '''
# 	Creates samples so each new example advances by 1 timestep (1 new label example)
# '''
# def sample_creater(trace_list, count, time_steps, label_card, label_size = 1, offset = 0):
# 	if count + time_steps + label_size + offset > len(trace_list):
# 		raise ValueError("trace_list is too short."
# 						 "\tcount = " + str(count) +
# 						 "\tlabel_size = " + str(label_size) +
# 						 "\tlen(trace_list) = " + str(len(trace_list)) +
# 						 "\toffset = " + str(offset))
# 	if time_steps >= len(trace_list):
# 		raise ValueError("time_steps is larger than trace_list.\ntrace_list is too short"
# 			  "\ttime_steps = " + str(time_steps) +
# 			  "\tlen(trace_list)" + str(len(trace_list)))
#
# 	example = np.zeros((count, time_steps, label_card))
# 	label = np.zeros((count, time_steps, label_card))
#
# 	for l in range(count):
# 		for j in range(time_steps):
# 			''' TODO: find the correct index'''
# 			task_num = trace_list[l + j]
# 			example[l][j][task_num] = 1
# 			for k in range(1, label_size + 1):
# 				''' TODO: find the correct index'''
# 				task_num = trace_list[l + j + k + offset]
# 				label[l][j][task_num] = 1
# 	return example, label
#
# '''
# 	requires the length of the trace_list to be divisible by count.
# '''
# def sample_creater_disjoint(trace_list, count, time_steps, label_card, label_size = 1, offset = 0):
#
# 	example_len = (label_size + time_steps)
#
# 	example = np.zeros((count, time_steps, label_card))
# 	label = np.zeros((count, time_steps, label_card))
#
# 	for i in range(count):
# 		for j in range(time_steps):
# 			task_num = trace_list[i*(example_len) + j]
# 			example[i][j][task_num] = 1
# 			for k in range(1,1+label_size):
# 				label_num = trace_list[i*(example_len) + j + k + offset]
# 				label[i][j][label_num] = 1
#
# 	return example, label

# ''' DEBUG THIS'''
# def generate_folds(folds, trace_list, training_ratio, time_steps, label_card, label_size=1, offset = 0):
# 	trace_folds = []
# 	fold_size = math.floor(len(trace_list)/folds)
# 	for i in range(folds):
# 		trace_folds.append(trace_list[fold_size * i:fold_size*(i+1)])
#
# 	result = []
# 	for i in range(folds):
# 		train, _ =list_to_array(trace_folds[i], 1, 100, 16)
# 		result.append(train)
#
# 	return result

def generate_folds2(folds, data_pair):
	example_list = data_pair[0]
	label_list = data_pair[1]

	if len(example_list) != len(label_list):
		raise ValueError("expected example and label to have same number of samples")

	fold_size = math.floor(len(example_list)/folds)

	time_steps = example_list[0].shape[0]
	label_card = example_list[0].shape[1]
	label_size = label_list[0].shape[0]


	result = []
	for i in range(folds):
		example_dataset = np.zeros((fold_size, time_steps, label_card))
		example_fold = example_list[i*fold_size:(i+1)*fold_size]
		for j in range(len(example_fold)):
			example_dataset[j]=example_fold[j]

		label_dataset = np.zeros((fold_size, label_size, label_card))
		label_fold = label_list[i * fold_size : (i + 1) * fold_size]
		for j in range(len(label_fold)):
			label_dataset[j]=label_fold[j]

		result.append((example_dataset, label_dataset))
	return result

def index_of_label(vec):
	for i in range(len(vec)):
		if vec[i] == 1:
			return i
	return -1

''' currently only checks correctly when label_size = 1'''
def validate_dataset(example_dataset, label_dataset, offset=0):
	label_size = label_dataset[0].shape[0]
	print(label_size)
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






res = generate_folds2(5,list_to_example(text_to_list('dataset_1.txt'),16,100,2))
validate_dataset(res[0][0],res[0][1],0)
