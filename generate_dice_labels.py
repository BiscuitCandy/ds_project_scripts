import csv 
import numpy as np
import pandas as pd

data_columns = [str(i) for i in range(1, 769)]

def preprocess(data):

    data.dropna(inplace=True)

    for i in data_columns :
        data[i] = data[i].astype(float)

    return data

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    angle_in_radians = np.arccos(cos_theta)
    angle_in_degrees = np.degrees(angle_in_radians)
    
    return angle_in_degrees



def label_maker(range1, range2, vector1, vector2):
    # Define the ranges (replace these with actual values)
    range1_start, range1_end = map(float, range1.replace("[", "").replace("]", "").split("|"))
    range2_start, range2_end = map(float, range2.replace("[", "").replace("]", "").split("|"))

    range1V_start = vector1[:384]
    range1V_end = vector1[384:]

    range2V_start = vector2[:384]
    range2V_end = vector2[384:]

    # Equal Ranges
    if range1_start == range2_start and range1_end == range2_end:
        if range1V_start == range2V_start and range1V_end == range2V_end:
            label = 1
        else:
            label = 0

    # Non-Overlapping Ranges
    if range1_end < range2_start or range2_end < range1_start:
        if (angle_between_vectors(range1V_start, range2V_start) + angle_between_vectors(range1V_start, range2V_end) - angle_between_vectors(range2V_start, range2V_end)
        + angle_between_vectors(range1V_end, range2V_start) + angle_between_vectors(range1V_end, range2V_end) - angle_between_vectors(range2V_start, range2V_end)) > 0:
            label = 1
        else:
            label = 0

    # Overlapping Ranges
    if range1_start <= range2_end and range2_start <= range1_end:
        if (angle_between_vectors(range2V_start, range1V_start) + angle_between_vectors(range2V_start, range1V_end) - angle_between_vectors(range1V_start, range1V_end)
        + angle_between_vectors(range2V_end, range1V_start) + angle_between_vectors(range2V_end, range1V_end) - angle_between_vectors(range1V_start, range1V_end)) == 0:
            label = 1
        else:
            label = 0

    # Range 1 Subsets Range 2
    if range2_start <= range1_start and range2_end >= range1_end:
        if (angle_between_vectors(range1V_start, range2V_start) + angle_between_vectors(range1V_start, range2V_end) - angle_between_vectors(range2V_start, range2V_end)
        + angle_between_vectors(range1V_end, range2V_start) + angle_between_vectors(range1V_end, range2V_end) - angle_between_vectors(range2V_start, range2V_end)) == 0:
            label = 1
        else:
            label = 0

    # Range 2 Subsets Range 1
    if range1_start <= range2_start and range1_end >= range2_end:
        if (angle_between_vectors(range2V_start, range1V_start) + angle_between_vectors(range2V_start, range1V_end) - angle_between_vectors(range1V_start, range1V_end)
        + angle_between_vectors(range2V_end, range1V_start) + angle_between_vectors(range2V_end, range1V_end) - angle_between_vectors(range1V_start, range1V_end)) == 0:
            label = 1
        else:
            label = 0

    # Range 1 Contains Range 2 (Partial)
    if range1_start <= range2_start and range1_end >= range2_start and range1_end <= range2_end:
        if (angle_between_vectors(range2V_start, range1V_start) + angle_between_vectors(range2V_start, range1V_end) - angle_between_vectors(range1V_start, range1V_end)
        + angle_between_vectors(range1V_end, range2V_start) + angle_between_vectors(range1V_end, range2V_end) - angle_between_vectors(range2V_start, range2V_end)) == 0:
            label = 1
        else:
            label = 0

    # Range 2 Contains Range 1 (Partial)
    if range2_start <= range1_start and range2_end >= range1_start and range2_end <= range1_end:
        if (angle_between_vectors(range1V_start, range2V_start) + angle_between_vectors(range1V_start, range2V_end) - angle_between_vectors(range2V_start, range2V_end)
        + angle_between_vectors(range2V_end, range1V_start) + angle_between_vectors(range2V_end, range1V_end) - angle_between_vectors(range1V_start, range1V_end)) == 0:
            label = 1
        else:
            label = 0

    # Range 1 Before Range 2
    if range1_end < range2_start:
        if (angle_between_vectors(range1V_start, range2V_start) + angle_between_vectors(range1V_start, range2V_end) - angle_between_vectors(range2V_start, range2V_end)
        + angle_between_vectors(range1V_end, range2V_start) + angle_between_vectors(range1V_end, range2V_end) - angle_between_vectors(range2V_start, range2V_end)) > 0:
            label = 1
        else:
            label = 0

    # Range 1 After Range 2
    if range1_start > range2_end:
        if (angle_between_vectors(range1V_start, range2V_start) + angle_between_vectors(range1V_start, range2V_end) - angle_between_vectors(range2V_start, range2V_end)
        + angle_between_vectors(range1V_end, range2V_start) + angle_between_vectors(range1V_end, range2V_end) - angle_between_vectors(range2V_start, range2V_end)) > 0:
            label = 1
        else:
            label = 0

    return label

dice_ranges = pd.read_csv("cancerkg2_meta_v_dice.tsv", sep = "\t", names = data_columns)

dice_numerics = open("../meta_v_output", 'r').readlines()

dice_ranges = preprocess(dice_ranges)

dice_ranges["vectors"] = dice_ranges[data_columns].apply(list, axis=1)

dice_ranges = dice_ranges["vectors"]

vectors = dice_ranges.tolist()
vectors = vectors[: min(len(dice_numerics), len(vectors))]
dice_numerics = dice_numerics[: min(len(dice_numerics), len(vectors))]

data = open("meta_v_dice_labels.tildesv", 'w+')

data_writer = csv.writer(data, delimiter="~")
data_writer.writerow(["range1", "range2", "vector1", "vector2", "label"])

for i in range(len(dice_numerics)- 1):
    range_1 = dice_numerics[i]
    range_2 = dice_numerics[i+1]

    vector1 = vectors[i]
    vector2 = vectors[i+1]

    res = (label_maker(range_1, range_2, vector1, vector2))
    
    data_writer.writerow([range_1, range_2, vector1, vector2, res])

print("Com[ple]teD")