from os import listdir
from os.path import isfile, join

# returns a list of names that have a result but does not have corresponding data
# i.e. returns set(dir1) - set(dir2)
def directory_set_minus(dir1), dir2):
    elems_in_dir1 = [f for f in listdir(dir1)]
    elems_in_dir2 = [f for f in listdir(dir2)]
    dir1_set = set(elems_in_dir1)
    dir2_set = set(elems_in_dir2)
    return dir1_set - dir2_set
