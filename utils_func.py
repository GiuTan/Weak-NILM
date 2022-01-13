import numpy as np
from sklearn.metrics import hamming_loss, precision_recall_curve, classification_report, roc_curve, auc
from matplotlib import pyplot as plt
from itertools import cycle
import scipy.signal
import random

def standardize_data(agg, mean, std):
    agg= agg -  mean
    agg /= std
    return agg

def output_binarization(output,thres):
    new_output = []
    for i in range(len(output)):
        matrix = np.zeros((2550, 5))
        for l in range(len(output[i])):
            curr = output[i]
            for k in range(5):
                    if curr[l][k] >= thres:
                        curr[l][k] = 1
                    else:
                        if curr[l][k] == -1:
                            curr[l][k] = -1
                        else:
                            curr[l][k] = 0
            matrix[l] = curr[l]
        new_output.append(matrix)

    new_output = np.array(new_output)
    return new_output

def app_binarization_weak(output,thres, classes):
    new_output = []
    for i in range(classes):
            for k in range(len(output)):
                #curr = output[k]
                if output[k][i] >= thres[i]:
                    output[k][i] = 1
                else:
                    output[k][i] = 0
                #matrix[l] = curr[l]
                #new_output.append(matrix)

    # new_output = np.array(new_output)
    # return new_output
    return output

def app_binarization_strong(output,thres, classes):
    new_output = []
    for k in range(len(output)):
             for i in range(classes):
                #curr = output[k]
                if output[k][i] >= thres[i]:
                    output[k][i] = 1
                else:
                    output[k][i] = 0
                #matrix[l] = curr[l]
             new_output.append(output[k])

    new_output = np.array(new_output)
    # return new_output
    return new_output

def thres_analysis(Y_test,new_output,classes):

    precision = dict()
    recall = dict()
    thres_list_strong = []

    for i in range(classes):

        precision[i], recall[i], thresh = precision_recall_curve(Y_test[:, i], new_output[:, i])

        plt.title('Pres-Recall-THRES curve')
        plt.plot(precision[i], recall[i])
        plt.show()
        plt.close()

        f1 = (2 * precision[i] * recall[i] )/ (precision[i] + recall[i])
        opt_thres_f1 = np.argmax(f1)
        optimal_threshold_f1 = thresh[opt_thres_f1]
        print("Threshold for F1-SCORE value is:", optimal_threshold_f1)
        if optimal_threshold_f1 >= 0.955:
             optimal_threshold_f1 = 0.55
        thres_list_strong.append(optimal_threshold_f1)

    return thres_list_strong


def weak_count(Y_train_weak):
    list_counter = [0, 0, 0, 0, 0]

    for i in range(len(Y_train_weak)):
        vect = Y_train_weak[i]
        for k in range(5):
            if vect[0][k] == 1:
                list_counter[k] += 1
    print("Weak composition:",list_counter)

