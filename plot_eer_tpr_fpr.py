import numpy as np
from eer_calculation import cal_metric


def graph_show(spoof_score_txt):
    
    with open(spoof_score_txt, 'r') as f:
        spoof_score = np.array(f.read().splitlines(), dtype=np.float)
    print(spoof_score.shape)

    # test set has number live sample : 20955
    # test set has number spoof sample : 23437
    labels = np.array([0]*20955 + [1]*23437, dtype=np.float)

    result_spoof = cal_metric(labels, spoof_score)
    print('eer spoof is : ' + str(result_spoof[0]) )
    print('tpr spoof is : ' + str(result_spoof[1]) )
    print('auc spoof is : ' + str(result_spoof[2]) )
    print('threshold for eer is : ' + str(result_spoof[4]) )

    # class_predcit = np.round(spoof_score)
    class_predict = np.array(np.where(spoof_score < result_spoof[4], 0, 1))

    temp = 0
    for i in range(labels.shape[0]):
        if class_predict[i] == labels[i]:
            temp += 1

    acc = round(temp/labels.shape[0], 4)
    print("acc of model is: ", acc)


if __name__ == '__main__':
    # spoof_score_txt = '/home/duong/project/pyimage_research/version2_change_data/result_b1_ver01/test_flow_from_directory_cp_06/score_prediction.txt'
    # spoof_score_txt = '/home/duong/project/pyimage_research/version2_change_data/result_new_b4_ver01/test_flow_from_directory_cp_04/score_prediction.txt'
    spoof_score_txt = '/home/duong/project/pyimage_research/version2_change_data/result_new_b0_ver3/test_flow_from_directory_cp_06/score_prediction.txt'
    graph_show(spoof_score_txt)