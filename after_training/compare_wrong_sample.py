"""
this script take 2 files which contain wrong sample in test folder
I have wonder if there some common image that models predict wrong
so this scipt will show joint sample
"""


import os


def show_number_same_wrong(file_1, file_2, file_3=None):
    # file 1 and file 2 is txt type
    # first step is read from these files
    if not file_3:
        with open(file_1) as f:
            wrong_list_1 = f.read().splitlines()
            print(f"file 1 has number of wrong: {len(wrong_list_1)} ")
        with open(file_2) as f:
            wrong_list_2 = f.read().splitlines()
            print(f"file 2 has number of wrong: {len(wrong_list_2)} ")

        same_number = 0
        same_list = []
        for item in wrong_list_1:
            if item in wrong_list_2:
                same_number += 1
                same_list.append(item)
        print(f"two file has number of same wrong is : {same_number}")
        return same_number, same_list
    else:
        with open(file_1) as f:
            wrong_list_1 = f.read().splitlines()
            print(f"file 1 has number of wrong: {len(wrong_list_1)} ")
        with open(file_2) as f:
            wrong_list_2 = f.read().splitlines()
            print(f"file 2 has number of wrong: {len(wrong_list_2)} ")
        with open(file_3) as f:
            wrong_list_3 = f.read().splitlines()
            print(f"file 3 has number of wrong: {len(wrong_list_3)} ")
        same_number = 0
        same_list = []
        for item in wrong_list_1:
            if item in wrong_list_2 and item in wrong_list_3:
                same_number += 1
                same_list.append(item)
        print(f"three file has number of same wrong is : {same_number}")
        return same_number, same_list

def print_list(l:list):
    for item in l:
        print(item)


file_1 = '/home/duong/project/pyimage_research/version2_change_data/result_new_b0_ver4/test_flow_from_directory_cp_01/wrong_sample_path.txt'
file_2 = '/home/duong/project/pyimage_research/version2_change_data/result_new_b0_ver4/test_flow_from_directory_cp_02/wrong_sample_path.txt'
file_3 = '/home/duong/project/pyimage_research/version2_change_data/result_new_b0_ver4/test_flow_from_directory_cp_03/wrong_sample_path.txt'
same_number,same_list = show_number_same_wrong(file_1, file_2, file_3)

print_list(same_list)


