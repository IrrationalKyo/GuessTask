import math
import glob
import matplotlib.pyplot as plt
import numpy as np
import json


# Used in parse_meta(.)
def get_value_in_paren(input_str):
    left_index = input_str.index("(")
    right_index = input_str.index(")")
    return input_str[left_index + 1:right_index]


def get_task_util_factor(task_dict):
    return float(task_dict["c"]) / task_dict["p"]


def parse_old_stat(file_name):
    statDict = {}

    with open(file_name, "r") as statFile:
        line = statFile.readline()
        while line:
            '''
                bunch of switch statements.
            '''
            typeSplit = line.split(":")
            if typeSplit[0] == "accuracy":
                statDict["accuracy"] = float(typeSplit[1].strip())
            else:
                taskNum = int(typeSplit[0].split("_")[1].strip())
                statDict[taskNum] = float(typeSplit[1].strip())
            line = statFile.readline()

    return statDict


# returns list of task_struct which has id, executiontime and etc.
def parse_meta(fileName):
    totalUtilityFactor = -1
    metaList = []
    with open(fileName,"r") as metaFile:
        line = metaFile.readline()
        while line:
            typeSplit = line.split(":")
            if "TaskSet(" in typeSplit[0]:
                totalUtilityFactor = float(get_value_in_paren(typeSplit[0]))
            else:
                taskDict = {}
                taskNum = int(typeSplit[0].split("-")[1])
                taskDict["id"] = taskNum
                propertySplit = typeSplit[1].split(",")
                for prop in propertySplit:
                    type = prop.split("=")[0].strip()
                    value = int(prop.split("=")[1])
                    taskDict[type] = value
                metaList.append(taskDict)

            line = metaFile.readline()
    average = 0
    for taskDict in metaList:
        average += get_task_util_factor(taskDict)
    average /= len(metaList)
    variance = 0
    for taskDict in metaList:
        utilFactor = get_task_util_factor(taskDict)
        variance += math.pow((utilFactor-average),2)
    variance /= len(metaList)
    return totalUtilityFactor, variance, metaList


def old_stat_to_json(file_name):
    json_dict = {}

    with open(file_name, "r") as file:
        for line in file:
            field_and_value = line.split(":")
            field = field_and_value[0]
            value = float(field_and_value[1])
            json_dict[field] = value

    parsed_json = json.dumps(json_dict)

    with open(file_name + ".json", "w") as file:
        file.write(parsed_json)
    return


def stat_to_json(file_name):
    return json.load(open(file_name,"r"))


def is_stat_up_to_date(file_name):
    stat_json = stat_to_json(file_name)
    if "total_utility_factor" not in stat_json or \
            "utility_variance" not in stat_json or \
            "accuracy" not in stat_json:
        return False
    return True


# Generates stat given meta_file and confusion matrix
def generate_stat(output_file_name, meta_file_name,cnf_matrix, accuracy):
    json_dict = {}
    total_utility_factor, utility_variance, meta_list = parse_meta(meta_file_name)
    json_dict["total_utility_factor"] = total_utility_factor
    json_dict["utility_variance"] = utility_variance
    json_dict["accuracy"] = {}
    json_dict["accuracy"]["overall"] = accuracy
    for i in range(len(cnf_matrix)):
        json_dict["accuracy"][i] = cnf_matrix[i][i]
    return json_dict


if __name__ == "__main__":

    # print(parseStat("./result/size26rep0.result/stat"))
    metaDict = {}
    statDict = {}
    for metaFile in glob.glob("./metas/*_meta.txt"):
        metaId = metaFile.split("/")[-1].split("_meta.txt")[0]
        metaDict[metaId] = parse_meta(metaFile)

    for resultFile in glob.glob("./result/*.result/stat"):
        metaId = resultFile.split("/")[-2].split(".result")[0]
        statDict[metaId] = parse_old_stat(resultFile)



    '''
        Utilization variance VS accuracy
    '''

    scatterX = np.zeros(len(statDict))
    scatterY = np.zeros(len(statDict))

    i = 0
    for key, values in statDict.item():
        metaObj = metaDict[key]
        scatterX[i] = metaObj[1]
        scatterY[i] = values["accuracy"]
        i+=1


    fig, ax = plt.subplots()
    ax.scatter(scatterX, scatterY)
    plt.show()
