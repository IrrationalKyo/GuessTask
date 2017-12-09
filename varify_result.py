import converter as cvt
# import guesser as gue

if __name__ == "__main__":
    pure_raw_data = cvt.text_to_list('dataset_det_1.txt')
    raw_data = cvt.list_to_example(pure_raw_data,16,1)
    print(raw_data[0][0])
    print(raw_data[1][0])