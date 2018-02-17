from os import listdir
from os.path import isfile, join
import os
import copy as cp
import re

class DirectoryOrganizer:

    def __init__(self, directory):
        self.directory = directory
        self.dir_dict = self.directory_parse()


    @staticmethod
    def parse_name(filename):
        # delimits when number to letter or letter to number occurs
        print(filename)
        temp =  re.split('(\d+)',filename)
        tasksize = int(temp[1])
        rep_count = int(temp[3])
        return (tasksize, rep_count)


    @staticmethod
    def reconstruct_name(tasksize, rep_count):
        return "size" + str(tasksize) + "rep" + str(rep_count)


    @staticmethod
    def dict_to_name(dictionary):
        name_set = set()
        for tasksize, rep_set in dictionary.items():
            for rep_count in rep_set:
                name_set.add(reconstruct_name(tasksize, rep_count))
        return name_set


    # reads a directory and returns a dictionary(tasksize, list of names)
    def directory_parse(self):
        elems_in_directory = [self.parse_name(f) for f in listdir(self.directory)]
        dir_dict = {}
        for elem in elems_in_directory:
            if elem[0] not in dir_dict:
                dir_dict[elem[0]] = set([elem[1]])
                continue
            dir_dict[elem[0]].add(elem[1])
        return dir_dict


    # returns a list of names that have a result but does not have corresponding data
    # i.e. returns directory_dictionary
    def __sub__(self, other_dir):
        this_dir_dictionary = cp.deepcopy(self.dir_dict)
        other_dir_dictionary = other_dir.dir_dict
        for tasksize, rep_set in other_dir_dictionary.items():
            if tasksize not in this_dir_dictionary:
                continue
            this_dir_dictionary[tasksize] = this_dir_dictionary[tasksize] - rep_set

            # removes the key if the rep_set is empty
            if not this_dir_dictionary[tasksize]:
                this_dir_dictionary.pop(tasksize, None)

        return this_dir_dictionary


class ResultOrganizer(DirectoryOrganizer):

    def __init__(self, directory):
        super().__init__(directory)
        # TODO: filter out the ones without models
        # DONE
        for tasksize, rep_set in self.dir_dict.items():
            for rep_count in rep_set:
                if (not self.model_exist(tasksize, rep_count)) \
                    or (not self.stat_exist(tasksize, rep_count)):
                    rep_set.remove(rep_count)


    '''
        TODO: could refact name parsing in *_exist methods
    '''
    def model_exist(self, tasksize, rep_count = None):
        # only one argument was provided
        # which means that the tasksize is a string
        # therefore, convert to appropriate format
        if rep_count is None:
            name_string = tasksize
            tasksize, rep_count = taskself.parse_name(name_string)

        directory_string = self.directory + "/" \
            + DirectoryOrganizer.reconstruct_name(tasksize, rep_count) \
            +".result"

        if not os.path.exists(directory_string):
            return False

        for fname in os.listdir(directory_string):
            if fname.endswith(".model"):
                return True
        return False


    def stat_exist(self, tasksize, rep_count = None):
        # only one argument was provided
        # which means that the tasksize is a string
        # therefore, convert to appropriate format
        if rep_count is None:
            name_string = tasksize
            tasksize, rep_count = taskself.parse_name(name_string)

        directory_string = self.directory + "/" \
            + DirectoryOrganizer.reconstruct_name(tasksize, rep_count) \
            +".result"

        if not os.path.exists(directory_string):
            return False

        for fname in os.listdir(directory_string):
            if fname.endswith("stat.json"):
                return True
        return False

class DataResultOrganizer:

    def __init__(self, data_dir, result_dir):
        self.org_data = DirectoryOrganizer(data_dir)
        self.org_result = ResultOrganizer(result_dir)

    def get_unprocessed_data_name(self):
        output_string = [];
        for tasksize, rep_set in (self.org_data - self.org_result).items():
            for rep_count in rep_set:
                output_string.append(DirectoryOrganizer.reconstruct_name(tasksize, rep_count)+".data")
        return output_string
        

if __name__ == "__main__":
    # org_metas = DirectoryOrganizer("./metas")
    # org_result = ResultOrganizer("./result")

    drorg = DataResultOrganizer("./metas", "./result")
    print(drorg.get_unprocessed_data_name())
