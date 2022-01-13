import os
import numpy as np
from datetime import datetime
import argparse
import gc
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import precision_recall_curve, classification_report
from CRNN import *
from utils_func import *
import json
import random as python_random
import tensorflow as tf
from metrics import *

parser = argparse.ArgumentParser(description="UK-DALE synthetic dataset creation")

parser.add_argument("--quantity_1", type=int, default=45581, help="Number of bags in UKDALE house 1")
parser.add_argument("--quantity_2", type=int, default=3271, help="Number of bags in UKDALE house 2")
parser.add_argument("--quantity_3", type=int, default=3047, help="Number of bags in UKDALE house 3")
parser.add_argument("--quantity_4", type=int, default=553, help="Number of bags in UKDALE house 4")
parser.add_argument("--quantity_5", type=int, default=2969, help="Number of bags in UKDALE house 5")
parser.add_argument("--perc_strong", type= int, default=20, help="Percentage of UKDALE strong data")
parser.add_argument("--perc_weak", type= int, default=80, help="Percentage of REFIT weak data")
parser.add_argument("--test", type= bool, default=False, help="Flag to perform inference")
parser.add_argument("--test_ukdale", type=bool, default=True, help="Flag to select which dataset has to be used for testing")
parser.add_argument("--refit_synth", type=str, default='', help="REFIT synth data path")
parser.add_argument("--ukdale_synth", type=str, default='', help="UKDALE synth data path")
arguments = parser.parse_args()


if __name__ == '__main__':
    # UK-DALE path
    path = '../weak_labels/'
    file_agg_path = path + 'aggregate_data_noised/'
    file_labels_path = path + 'labels/'

    # REFIT path
    refit_agg_resample_path = '../resampled_agg_REFIT/'

    # set seeds for reproducible results
    random.seed(123)
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    quantity_1 = arguments.quantity_1
    quantity_2 = arguments.quantity_2
    quantity_5 = arguments.quantity_5
    quantity_4 = arguments.quantity_4
    quantity_3 = arguments.quantity_3
    houses = [1, 2, 3, 4, 5]
    houses_id = [0, 'house_1/', 'house_2/', 'house_3/', 'house_4/', 'house_5/']

    perc_strong = arguments.perc_strong
    print("perc strong", perc_strong)
    perc_weak = arguments.perc_weak
    print("perc weak:", perc_weak)

    # Flags Inizialization

    test = arguments.test
    strong_weak = True
    strong_weak_flag = True
    test_ukdale = arguments.test_ukdale
    weak_counter = True

    X_train, Y_train, Y_train_weak = [], [], []
    X_test, Y_test, Y_test_weak = [], [], []
    X_val, Y_val, Y_val_weak = [], [], []

    for k in houses:

        count_str = 0
        count_val = 0
        count_weak = 0


        f = open(file_labels_path + 'labels_%d.json' % k)
        labels = json.load(f)
        print("Labels Loaded")
        if k == 1:
            quantity = quantity_1
        if k == 2:
            quantity = quantity_2
        if k == 5:
            quantity = quantity_5
        if k == 3:
            quantity = quantity_3
        if k == 4:
            quantity = quantity_4

        b = round(quantity / 5)
        a = round(b / 5)

        for i in range(quantity):

            try:
                agg = np.load(file_agg_path + houses_id[k] + 'aggregate_%d.npy' % i)
            except FileNotFoundError:
                continue

            key = 'labels_%d' % i

            #  STRONG  #
            try:
                list_strong = labels[key]['strong']
            except KeyError:
                continue

            matrix = np.zeros((5, 2550))
            error_vectors = 0



            for l in range(len(list_strong)):
                matrix[l] = np.array(list_strong[l])

            if k == 1 or k == 5 or k == 3 or k == 4:

                if i < a or (i >= b and i < (a + b)) or (i >= (b * 2) and i < (b * 2 + a)) or (
                            i >= (b * 3) and i < (b * 3 + a)) or (i >= b * 4 and i < (b * 4 + a)):
                        # se rientra nei dati di validation rimangono strong

                    matrix = np.transpose(matrix)
                    X_val.append(agg)
                    Y_val.append(matrix)


                else:  # se va nei dati di train

                    quantity_train = round(quantity / 100 * 80)
                    num_data = round(quantity_train / 100 * (100 - perc_strong))
                    print("Quantity of data without strong labels:")
                    print(num_data)
                    count_str += 1
                    if count_str < num_data:
                        matrix = np.ones((2550, 5))
                        matrix = np.negative(matrix)
                        X_train.append(agg)
                        Y_train.append(matrix)

                    else:
                        matrix = np.transpose(matrix)
                        X_train.append(agg)
                        Y_train.append(matrix)

            if k == 2:
                matrix = np.transpose(matrix)
                X_test.append(agg)
                Y_test.append(matrix)

            ##### WEAK #####
            try:
                list_weak = labels[key]['weak']
            except KeyError:
                continue

            if k == 1 or k == 5 or k == 3 or k == 4:

                if i < a or (i >= b and i < (a + b)) or (i >= (b * 2) and i < (b * 2 + a)) or (
                            i >= (b * 3) and i < (b * 3 + a)) or (i >= b * 4 and i < (b * 4 + a)):

                    Y_val_weak.append(np.array(list_weak).reshape(1, 5))

                else:

                    if strong_weak:
                        num_data = 0
                        print(num_data)
                        count_weak += 1
                        if count_weak <= num_data:
                            list_weak = [-1, -1, -1, -1,
                                             -1]
                            Y_train_weak.append(np.array(list_weak).reshape(1, 5))

                        else:
                            Y_train_weak.append(np.array(list_weak).reshape(1, 5))

                    else:
                        Y_train_weak.append(np.array(list_weak).reshape(1, 5))
            if k == 2:
                Y_test_weak.append(np.array(list_weak).reshape(1, 5))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Y_train_weak = np.array(Y_train_weak)
    if not test_ukdale:
        X_test, Y_test, Y_test_weak = [], [], []
        refit_agg_resample_path_test = '../resampled_agg_REFIT_test/'
        refit_labels_resample_path_test = '../resampled_labels_REFIT_test/'
        
	houses = [4, 9, 15]

	# number of bags in each house can change based on the created dataset 
        for k in houses:

            quant = [0, 0, 0, 0, 12000, 0, 0, 0, 0, 9000, 0, 0, 0, 0, 0, 4000, 0, 0, 0, 0, 0, 0, 0]

            error_vectors = 0
            print("Aggregate Loading")

            for i in range(quant[k]):

                agg = np.load(refit_agg_resample_path_test + 'house_' + str(k) + '/aggregate_%d.npy' % i)
                labels_weak = np.load('../labels/' + 'house_' + str(
                    k) + '/weak_labels_%d.npy' % i, allow_pickle=True)
                labels_strong = np.load(
                    refit_labels_resample_path_test + 'house_' + str(k) + '/strong_labels_%d.npy' % i,
                    allow_pickle=True)
                val_q = round(quant[k] / 100 * 20)

                if k == 4 or k == 9 or k == 15:

                        matrix = np.transpose(labels_strong)
                        X_test.append(agg)
                        Y_test.append(matrix)
                        Y_test_weak.append(labels_weak.reshape(1, 5))

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        Y_test_weak = np.array(Y_test_weak)
    else:
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        Y_test_weak = np.array(Y_test_weak)


    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    Y_val_weak = np.array(Y_val_weak)
    Y_val = output_binarization(Y_val, 0.4)

    assert (len(X_val) == len(Y_val))
    assert (len(Y_val) == len(Y_val_weak))

    houses_weak = [2, 5, 7, 10, 12, 13, 16]
    weak_X_train_balanced, weak_Y_train_balanced, weak_Y_train_weak_balanced = [], [], []
    
    # number of bags can change based on the created dataset
    for k in houses_weak:
        quant = [0, 0, 20000, 0, 0, 20000, 0, 20000, 0, 0, 20000, 0, 15000, 3000, 0, 0, 5000]

        for i in range(quant[k]):

            print("Aggregate Loading")
            agg = np.load(refit_agg_resample_path + 'house_' + str(k) + '/aggregate_%d.npy' % i)

            labels_weak = np.load('../labels/' + 'house_' + str(
                k) + '/weak_labels_%d.npy' % i, allow_pickle=True)
            labels_strong = np.load('..labels/' + 'house_' + str(
                k) + '/strong_labels_%d.npy' % i, allow_pickle=True)


            matrix = np.ones((5, 2550))
            matrix = np.negative(matrix)


            print("TRAIN")
            matrix = np.transpose(matrix)
            weak_X_train_balanced.append(agg)
            weak_Y_train_balanced.append(matrix)
            weak_Y_train_weak_balanced.append(labels_weak.reshape(1, 5))


    weak_X_train_balanced = np.array(weak_X_train_balanced)
    weak_Y_train_balanced = np.array(weak_Y_train_balanced)
    weak_Y_train_weak_balanced = np.array(weak_Y_train_weak_balanced)

    Y_train_new = []
    X_train_new = []
    Y_train_weak_new = []
    # it is necessary to considered only the strong labeled data so
    for i in range(len(Y_train)):
        if np.all(Y_train[i][0] != -1):
            Y_train_new.append(Y_train[i])
            X_train_new.append(X_train[i])
            Y_train_weak_new.append(Y_train_weak[i])

    Y_train_new = np.array(Y_train_new)
    Y_train_weak_new = np.array(Y_train_weak_new)
    X_train_new = np.array(X_train_new)
    print("Y train strong shape", Y_train_new.shape)

    X_train = np.concatenate([X_train_new, weak_X_train_balanced], axis=0)
    Y_train = np.concatenate([Y_train_new, weak_Y_train_balanced], axis=0)
    Y_train_weak = np.concatenate([Y_train_weak_new, weak_Y_train_weak_balanced], axis=0)

    num_weak = round(len(Y_train_weak) / 100 * perc_weak)
    num_non_weak = round(len(Y_train_weak) / 100 * (100 - perc_weak))
    Y_train_weak = Y_train_weak[:num_weak]
    Y_train = Y_train[:num_weak]
    X_train = X_train[:num_weak]

    # this function return the quantity of bags that contain each appliance
    weak_count(Y_train_weak)

    print("Total x train", X_train.shape)
    print("Total Y train", Y_train.shape)
    print("Total Y train weak", Y_train_weak.shape)
    assert (len(Y_val) == len(Y_val_weak))
    assert (len(Y_train) == len(Y_train_weak))

    x_train = X_train
    y_strong_train = Y_train
    y_weak_train = Y_train_weak

    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    print("STRONG-WEAK")
    print(perc_strong)
    print("Mean train")
    print(train_mean)
    print("Std train")
    print(train_std)

    x_train = standardize_data(x_train, train_mean, train_std)
    X_val = standardize_data(X_val, train_mean, train_std)
    X_test = standardize_data(X_test, train_mean, train_std)

    batch_size = 64
    window_size = 2550
    drop = 0.1
    kernel = 5
    num_layers = 3
    gru_units = 64
    cs = False
    type_ = 'UKDALE_REFIT_NOISED_' + str(perc_weak) + 'weak_' + str(
        perc_strong) + 'strong_weak_'
    lr = 0.002
    drop_out = drop
    weight = 1e-2
    classes = 5

    CRNN = CRNN_construction(window_size, weight, lr=lr, classes=5, drop_out=drop, kernel=kernel, num_layers=num_layers,
                             gru_units=gru_units, cs=cs, strong_weak_flag=True)

    if cs:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_final_custom_f1_score', mode='max',
                                                      patience=15, restore_best_weights=True)
    else:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_custom_f1_score', mode='max',
                                                      patience=15, restore_best_weights=True)

    log_dir_ = '../logs_CRNN' + datetime.now().strftime(
        "%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    if not test:
        history = CRNN.fit(x=x_train, y=[y_strong_train, y_weak_train], shuffle=True, epochs=1000,
                           batch_size=batch_size,
                           validation_data=(X_val, [Y_val, Y_val_weak]), callbacks=[early_stop, tensorboard], verbose=1)
        CRNN.save_weights(
            '')

    else:
        CRNN.load_weights(
            '')

    output_strong, output_weak = CRNN.predict(x=X_val)
    output_strong_test_o, output_weak_test = CRNN.predict(x=X_test)
    print(Y_val.shape)
    print(output_strong.shape)

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]

    Y_val = Y_val.reshape(shape, 5)
    Y_test = Y_test.reshape(shape_test, 5)
    output_strong = output_strong.reshape(shape, 5)
    output_strong_test = output_strong_test_o.reshape(shape_test, 5)


    thres_strong = thres_analysis(Y_val, output_strong, classes)

    output_weak_test = output_weak_test.reshape(output_weak_test.shape[0] * output_weak_test.shape[1], 5)
    output_weak = output_weak.reshape(output_weak.shape[0] * output_weak.shape[1], 5)
    print(output_weak)
    thres_weak = [0.501, 0.501, 0.501, 0.501, 0.501]

    assert (Y_val.shape == output_strong.shape)

    plt.plot(output_strong[:24000, 0])
    plt.plot(Y_val[:24000, 0])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 1])
    plt.plot(Y_val[:24000, 1])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 2])
    plt.plot(Y_val[:24000, 2])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 3])
    plt.plot(Y_val[:24000, 3])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 4])
    plt.plot(Y_val[:24000, 4])
    plt.legend(['output', 'strong labels'])
    plt.show()

    print("Estimated best thresholds:", thres_strong)

    output_strong_test = app_binarization_strong(output_strong_test, thres_strong, 5)
    output_strong = app_binarization_strong(output_strong, thres_strong, 5)

    print("STRONG SCORES:")
    print("Validation")
    print(classification_report(Y_val, output_strong))
    print("Test")
    print(classification_report(Y_test, output_strong_test))

    if test_ukdale:
        houses = [2]
        synth_agg_path = arguments.ukdale_synth
        X_test_synth = []
        Y_test = []
        for k in houses:

            f = open(file_labels_path + 'labels_%d.json' % k)
            labels = json.load(f)
            print("Labels Loaded")
            if k == 1:
                quantity = quantity_1
            if k == 2:
                quantity = quantity_2
            if k == 5:
                quantity = quantity_5
            if k == 3:
                quantity = quantity_3
            if k == 4:
                quantity = quantity_4

            b = round(quantity / 5)
            a = round(b / 5)

            for i in range(quantity):

                agg = np.load(synth_agg_path + houses_id[k] + 'aggregate_%d.npy' % i)

                key = 'labels_%d' % i

                #  STRONG  #
                list_strong = labels[key]['strong']

                matrix = np.zeros((5, 2550))


                for l in range(len(list_strong)):
                    matrix[l] = np.array(list_strong[l])

                    matrix = np.transpose(matrix)
                    X_test_synth.append(agg)
                    Y_test.append(matrix)

            X_test_synth = np.array(X_test_synth)

            ANE = ANE(X_test_synth, output_strong_test_o)
            print("ANE UKDALE:", ANE)

    else:
        refit_synth_agg_resample_path = arguments.refit_synth
        houses_re_test = [4, 9, 15]
        quantity_9 = 9000
        quantity_15 = 1500
        quantity_4 = 12000
        X_test = []

        for k in houses_re_test:

            quant = [0, 0, 0, 0, quantity_4, 0, 0, 0, 0, quantity_9, 0, 0, 0, 0, 0, quantity_15, 0, 0, 0, 0]

            for i in range(quant[k]):

                agg = np.load(refit_synth_agg_resample_path + 'house_' + str(k) + '/aggregate_%d.npy' % i)


                X_test.append(agg)

            X_test_synth = np.array(X_test)
            ANE = ANE(X_test_synth, output_strong_test_o)
            print("ANE REFIT:", ANE)

