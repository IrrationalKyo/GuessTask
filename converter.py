from keras.datasets import imdb
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
	data_size = math.floor(len(trace_list)/(time_steps + label_size))
	# training_size := number of training samples
	training_size = int(data_size * training_ratio)
	# test_size := number of test samples
	test_size = data_size - training_size


	training_examples = np.zeros((training_size, time_steps, label_card))
	training_labels = np.zeros((training_size, time_steps, label_card))

	test_examples = np.zeros((test_size, time_steps, label_card))
	test_labels = np.zeros((test_size, time_steps, label_card))
	
	for i in range(data_size):
		# if data_size - i < time_steps + label_size:
		# 	break
		if i < training_size:
			for j in range(time_steps):
				''' TODO: find the correct index'''
				task_num = trace_list[i + j]
				training_examples[i][j][task_num] = 1
				for k in range(label_size):
					''' TODO: find the correct index'''
					label = trace_list[i + time_steps + j]
					training_labels[i][j][label] = 1
		else:
			for j in range(time_steps):
				test_examples[i-training_size][0][j] = trace_list[i+j]
			for j in range(label_size):
				label = trace_list[i + time_steps + j]
				test_labels[i-training_size][0][label] = 1
	return (training_examples, training_labels), (test_examples,test_labels)
	

#(train_x, train_y), (test_x,test_y) = list_to_array(text_to_list('dataset_1.txt'), 0.8, 1000, 16)
#print(len(train_x))

#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
#print("train_x shape:" + str(X_train.shape))
#print("train_y shape:" + str(y_train.shape))