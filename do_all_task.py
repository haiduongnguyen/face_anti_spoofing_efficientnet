def show_image_size_survey():
    """
    task 1 : show result of survey image size
    """
    import os 
    import cv2
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import sys


    def show_graph(d:dict):
        """
        this function read data from a python dict
        then draw a graph to visualize data in dict
        """
        x = []
        y = []
        for key, value in d.items():
            x.append(str(key))
            y.append(value)

        x_pos = [i for i, _ in enumerate(x)]
        plt.figure()
        plt.bar(x_pos, y, color='green')
        plt.xlabel("Size")
        plt.ylabel("Number of images")
        plt.title("Count by size = width + height ")
        plt.xticks(x_pos, x)

    size_train= {100: 1077, 200: 17298, 300: 36400, 400: 42121, 500: 39558, 600: 34338, 700: 19799, 800: 33470}
    size_test = {100: 219, 200: 3483, 300: 6772, 400: 6289, 500: 7343, 600: 5800, 700: 4109, 800: 10378}

    show_graph(size_train)
    show_graph(size_test)
    plt.show()


def show_tpr_fpr_graph(spoof_score_txt):
    """
    spoof_score_txt is a file being generated by run evaluate
    this function take the path as input, and show result to screen
    """
    import numpy as np
    import os
    from eer_calculation import cal_metric
    with open(spoof_score_txt, 'r') as f:
        spoof_score = np.array(f.read().splitlines(), dtype=np.float)
    # print(spoof_score.shape)

    # test set has number live sample : 20955
    # test set has number spoof sample : 23438
    count_live = 20955
    count_spoof = 23438
    # count_spoof = 23437

    labels = np.array([0]*count_live + [1]*count_spoof, dtype=np.float)

    result_spoof = cal_metric(labels, spoof_score)
    print('eer spoof is : ' + str(result_spoof[0]) )
    print('tpr spoof is : ' + str(result_spoof[1]) )
    print('auc spoof is : ' + str(result_spoof[2]) )
    print('threshold for eer is : ' + str(result_spoof[4]) )

    # class_predcit = np.round(spoof_score)
    threshold_spoof = result_spoof[4]
    # threshold = 0.5
    class_predict = np.array(np.where(spoof_score < threshold_spoof, 0, 1))

    temp = 0
    for i in range(labels.shape[0]):
        if class_predict[i] == labels[i]:
            temp += 1

    acc = round(temp/labels.shape[0], 4)
    print(f"acc of model at threshold {threshold_spoof} is {acc}: ")


def get_input_shape_model():
    from keras.models import load_model
    import numpy as np
    model_path = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b0_ver4/cp_01.h5'
    my_model = load_model(model_path)
    input_model = my_model.input_shape
    width , height = input_model[1], input_model[2]
    print(width, height)
    # print(input_model.shape)


if __name__ == '__main__':
    ## task 1: survey in image size
    # show_image_size_survey()

    # task 2: show fnr, tpr, threshold graph
    # spoof_score_txt = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b0_ver4/test_flow_from_directory_cp_01/score_prediction.txt'
    # spoof_score_txt = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b4_ver01/test_flow_from_directory_cp_04/score_prediction.txt'
    # spoof_score_txt = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b1_ver1/test_tf_cp_15/score_prediction.txt'
    # show_tpr_fpr_graph(spoof_score_txt)


    ## task 3: load model and get input shape of model
    # get_input_shape_model()


    import os
    folder_path = '/home/duong/Desktop/test_spoof_card'
    count = 0
    for img_name in os.listdir(folder_path):
        if 'eval' in img_name:
            os.remove(os.path.join(folder_path, img_name))
            count += 1
    print(count)
