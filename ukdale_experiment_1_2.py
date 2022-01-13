from CRNN import *
import numpy as np
from utils_func import *
import json
import random as python_random
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from metrics import *
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description="UK-DALE synthetic dataset creation")

parser.add_argument("--quantity_1", type=int, default=45581, help="Number of bags in UKDALE house 1")
parser.add_argument("--quantity_2", type=int, default=3271, help="Number of bags in UKDALE house 2")
parser.add_argument("--quantity_3", type=int, default=3047, help="Number of bags in UKDALE house 3")
parser.add_argument("--quantity_4", type=int, default=553, help="Number of bags in UKDALE house 4")
parser.add_argument("--quantity_5", type=int, default=2969, help="Number of bags in UKDALE house 5")
parser.add_argument("--perc_strong", type= int, default=20, help="Percentage of UKDALE strong data")
parser.add_argument("--perc_weak", type= int, default=80, help="Percentage of UKDALE weak data")
parser.add_argument("--control_strong", type= bool, default=True, help="Flag to control if the correct quantity of strongly annotated bags have been considered")
parser.add_argument("--cs", type= bool, default=False, help="Clip smoothing post-processing")
parser.add_argument("--weak_houses", type=bool, default=True, help="The flag to load weak dataset")
parser.add_argument("--strong_weak_flag", type=bool, default=True, help="The flag to choose Strong-CRNN or Weak-CRNN")
parser.add_argument("--synth_path", type=str, default='', help="Path where test synth aggregate bags are stored")
parser.add_argument("--test", type=bool, default=False,help="Flag for inference")
arguments = parser.parse_args()


if __name__ == '__main__':


    file_agg_path = '../aggregate_data_noised/'
    file_labels_path = '../labels/'

    random.seed(123)
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


    quantity_1 = arguments.quantity_1
    quantity_2 = arguments.quantity_2
    quantity_5 = arguments.quantity_5
    quantity_4 = arguments.quantity_4
    quantity_3 = arguments.quantity_3

    perc_strong = arguments.perc_strong
    print("perc strong:",perc_strong)

    test = arguments.test

    houses = [1,2,3,4,5]
    houses_id = [0, 'house_1/', 'house_2/', 'house_3/', 'house_4/', 'house_5/']


    X_train, Y_train, Y_train_weak = [], [], []
    X_test, Y_test, Y_test_weak = [], [], []
    X_val, Y_val, Y_val_weak = [], [], []

    # LOADING DATA FROM .JSON FOR LABELS STRONG AND WEAK AND .NPY FOR AGGREGATE   #

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
                agg = np.load(file_agg_path + houses_id[k] + 'aggregate_%d.npy' %i)
            except FileNotFoundError:
                continue

            key = 'labels_%d' %i

            #  STRONG  #
            try:
                list_strong = labels[key]['strong']
            except KeyError:
                continue
            matrix = np.zeros((5, 2550))



            for l in range(len(list_strong)):
                matrix[l] = np.array(list_strong[l])


            if k == 1 or k == 5 or k==3 or k==4:

                if i < a or (i>=b and i <(a+b)) or (i>=(b*2) and i<(b*2 + a)) or (i>= (b*3) and i<(b*3 + a)) or (i>= b*4 and i<(b*4 + a)):


                            # Validation data are always annotated with both strong and weak labels
                            matrix = np.transpose(matrix)
                            X_val.append(agg)
                            Y_val.append(matrix)


                else:

                        quantity_train = round(quantity/100*80)
                        num_data = round(quantity_train / 100 * (100 - perc_strong))
                        print("Quantity of data without strong labels:")
                        print(num_data)
                        count_str += 1
                        if count_str < num_data:
                            matrix = np.ones((2550,5))
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




            if k == 1 or k == 5 or k==3 or k==4:

                if i < a or (i >= b and i < (a + b)) or (i >= (b * 2) and i < (b * 2 + a)) or (
                            i >= (b * 3) and i < (b * 3 + a)) or (i >= b * 4 and i < (b * 4 + a)):


                    Y_val_weak.append(np.array(list_weak).reshape(1,5))

                else:


                    Y_train_weak.append(np.array(list_weak).reshape(1, 5))
            if k == 2:

                Y_test_weak.append(np.array(list_weak).reshape(1,5))


    weak_X_train, weak_Y_train,weak_Y_train_weak   = [], [], []
    weak_X_train_balanced, weak_Y_train_balanced,weak_Y_train_weak_balanced   = [], [], []

    weak_houses = arguments.weak_houses
    print("Weak dataset loading:",weak_houses)
    if weak_houses:
        n_1 = 0
        f_weak = open(file_labels_path + 'labels_1_weak.json')
        labels_weak = json.load(f_weak)
        print("Labels weak Loaded")

        for i in range(len(labels_weak)):

            try:
                agg = np.load(file_agg_path + 'house_1_weak/' + 'aggregate_%d.npy' %i)
            except FileNotFoundError:
                continue


            key = 'labels_%d' %i

            #  STRONG  #
            try:
                list_strong = labels_weak[key]['strong']
            except KeyError:
                continue

            matrix = np.negative(np.ones((5, 2550)))
            weak_X_train.append(agg)
            weak_Y_train.append(matrix)
            ##### WEAK #####
            try:
                list_weak = labels_weak[key]['weak']
            except KeyError:
                continue

            weak_Y_train_weak.append(np.array(list_weak).reshape(1,5))



        f_weak_balanced = open(file_labels_path + 'labels_1_weak_balanced.json')
        labels_weak_balanced = json.load(f_weak_balanced)
        print("Labels weak balanced Loaded")

        for i in range(len(labels_weak_balanced)):

            try:
                agg = np.load('../aggregate_data_noised/house_1_weak_balanced/' + 'aggregate_%d.npy' % i)
            except FileNotFoundError:
                continue


            key = 'labels_%d' % i

            #  STRONG  #
            try:
                list_strong = labels_weak_balanced[key]['strong']
            except KeyError:
                continue

            matrix = np.negative(np.ones((2550, 5)))

            weak_X_train_balanced.append(agg)
            weak_Y_train_balanced.append(matrix)

            ##### WEAK #####
            try:
                list_weak = labels_weak_balanced[key]['weak']
            except KeyError:
                continue

            weak_Y_train_weak_balanced.append(np.array(list_weak).reshape(1, 5))

        print("Weak dataset loaded!")

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        Y_train_weak = np.array(Y_train_weak)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        Y_test_weak = np.array(Y_test_weak)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        Y_val_weak = np.array(Y_val_weak)
        print(Y_val_weak.shape)
        print(Y_val.shape)
        weak_X_train = np.array(weak_X_train)
        weak_Y_train = np.array(weak_Y_train)
        weak_Y_train_weak = np.array(weak_Y_train_weak)
        weak_X_train_balanced = np.array(weak_X_train_balanced)
        weak_Y_train_balanced = np.array(weak_Y_train_balanced)
        weak_Y_train_weak_balanced = np.array(weak_Y_train_weak_balanced)

        X_train = np.concatenate([X_train,weak_X_train, weak_X_train_balanced], axis=0)
        Y_train = np.concatenate([Y_train,weak_Y_train, weak_Y_train_balanced], axis = 0)
        Y_train_weak = np.concatenate([Y_train_weak,weak_Y_train_weak, weak_Y_train_weak_balanced], axis = 0)
        print(X_train.shape)
        print(Y_train.shape)
        print(Y_train_weak.shape)


    else:
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        Y_train_weak = np.array(Y_train_weak)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        Y_test_weak = np.array(Y_test_weak)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        Y_val_weak = np.array(Y_val_weak)
        print(Y_val_weak.shape)
        print(Y_val.shape)


    assert(len(Y_val)==len(Y_val_weak))
    assert(len(Y_train)==len(Y_train_weak))

    control_strong = arguments.control_strong
    if control_strong:
        Y_val_new = []
        X_val_new = []
        Y_val_weak_new = []
        Y_train_new = []
        X_train_new = []
        Y_train_weak_new = []

        for i in range(len(Y_val)):
            if np.all(Y_val[i][0] != -1):
                Y_val_new.append(Y_val[i])
                X_val_new.append(X_val[i])
                Y_val_weak_new.append(Y_val_weak[i])

        for i in range(len(Y_train)):
            if np.all(Y_train[i][0] != -1):
                Y_train_new.append(Y_train[i])
                X_train_new.append(X_train[i])
                Y_train_weak_new.append(Y_train_weak[i])

        Y_val_new = np.array(Y_val_new)
        Y_val_weak_new = np.array(Y_val_weak_new)
        X_val_new = np.array(X_val_new)
        Y_train_new = np.array(Y_train_new)
        Y_train_weak_new = np.array(Y_train_weak_new)
        X_train_new = np.array(X_train_new)
        print("Val strong shape")
        print(Y_val_new.shape)
        print("Train strong shape")
        print(Y_train_new.shape)




    x_train = X_train
    y_strong_train = Y_train
    y_weak_train = Y_train_weak

    print('Data shape:')
    print(x_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    # Aggregate Standardization #
    if perc_strong <= 100 and weak_houses == False:
        train_mean = np.mean(X_train_new)
        train_std = np.std(X_train_new)
        print(perc_strong)
        print("Mean train")
        print(train_mean)
        print("Std train")
        print(train_std)
    else:
        train_mean = np.mean(x_train)
        train_std = np.std(x_train)
        print(perc_strong)
        print("Mean train")
        print(train_mean)
        print("Std train")
        print(train_std)

    x_train = standardize_data(x_train,train_mean, train_std)
    X_val = standardize_data(X_val, train_mean, train_std)
    X_test = standardize_data(X_test,train_mean, train_std)
    perc_weak = arguments.perc_weak
    if weak_houses:

        len_tot_weak = len(Y_train_weak)
        quant_weak = round(len_tot_weak * perc_weak / 100)

        print("Weak quantity:", quant_weak)
        no_weak = np.ones((1, 5))
        no_weak = np.negative(no_weak)
        new_weak = []
        for lun in range(len_tot_weak):
            if lun <= quant_weak:
               new_weak.append(Y_train_weak[lun])
            else:
               new_weak.append(no_weak)

        Y_train_weak = np.array(new_weak)
        weak_count(Y_train_weak)
    type_ = 'NOISED_BEST_'+ str(perc_weak) + 'weak_' + str(perc_strong) + 'strong_'

    batch_size = 64
    window_size = 2550
    drop = 0.1
    kernel = 5
    num_layers = 3
    gru_units = 64
    lr = 0.002
    drop_out = drop
    weight= 1e-2

    CRNN = CRNN_construction(window_size,weight, lr=lr, classes=5, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=arguments.cs,strong_weak_flag=arguments.strong_weak_flag)

    if arguments.cs:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_final_custom_f1_score', mode='max',
                                                      patience=15, restore_best_weights=True)
    else:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_custom_f1_score', mode='max',
                                                      patience=15, restore_best_weights=True)


    log_dir_ = '/home/eprincipi/Weak_Supervision/weak_labels/models/logs/logs_CRNN'  + datetime.now().strftime("%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    if not test:
        if arguments.strong_weak_flag:
            history = CRNN.fit(x=x_train, y=[y_strong_train, y_weak_train], shuffle=True, epochs=1000,
                               batch_size=batch_size,
                               validation_data=(X_val, [Y_val, Y_val_weak]), callbacks=[early_stop, tensorboard], verbose=1)
            CRNN.save_weights(
                '')
            output_strong, output_weak = CRNN.predict(x=X_val)
            output_strong_test_o, output_weak_test = CRNN.predict(x=X_test)
        else:
            history = CRNN.fit(x=x_train, y=y_strong_train, shuffle=True, epochs=1000, batch_size=batch_size,
                               validation_data=(X_val, Y_val), callbacks=[early_stop], verbose=1)
            CRNN.save_weights(
                '')
            output_strong = CRNN.predict(x=X_val)
            output_strong_test_o = CRNN.predict(x=X_test)

    else:
        CRNN.load_weights('')

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]


    Y_val = Y_val.reshape(shape, 5)
    Y_test = Y_test.reshape(shape_test, 5)

    output_strong = output_strong.reshape(shape, 5)

    output_strong_test = output_strong_test_o.reshape(shape_test, 5)

    thres_strong = thres_analysis(Y_val, output_strong,classes=5)
    assert (Y_val.shape == output_strong.shape)


    plt.plot(output_strong[:24000, 0])
    plt.plot(Y_val[:24000, 0])
    plt.legend(['output', 'ground truth'])
    plt.show()
    plt.plot(output_strong[:24000, 1])
    plt.plot(Y_val[:24000, 1])
    plt.legend(['output', 'ground truth'])
    plt.show()

    plt.plot(output_strong[:24000, 2])
    plt.plot(Y_val[:24000, 2])
    plt.legend(['output', 'ground truth'])
    plt.show()

    plt.plot(output_strong[:24000, 3])
    plt.plot(Y_val[:24000, 3])
    plt.legend(['output', 'ground truth'])
    plt.show()

    plt.plot(output_strong[:24000, 4])
    plt.plot(Y_val[:24000, 4])
    plt.legend(['output', 'ground truth'])
    plt.show()

    print(thres_strong)


    output_strong_test = app_binarization_strong(output_strong_test, thres_strong, 5)
    output_strong = app_binarization_strong(output_strong, thres_strong, 5)


    print("STRONG SCORES:")
    print("Validation")

    print(classification_report(Y_val, output_strong))

    print("Test")

    print(classification_report(Y_test, output_strong_test))

    houses = [2]
    X_test = []
    Y_test = []
    synth_path = arguments.synth_path
    for k in houses:

        f = open(synth_path + 'labels_%d.json' % k)
        labels = json.load(f)
        print("Labels Loaded")

        if k == 2:
            quantity = quantity_2

        b = round(quantity / 5)
        a = round(b / 5)

        for i in range(quantity):

            agg = np.load(file_agg_path + houses_id[k] + 'aggregate_%d.npy' % i)

            key = 'labels_%d' % i

            #  STRONG  #
            list_strong = labels[key]['strong']

            matrix = np.zeros((5, 2550))


            for l in range(len(list_strong)):
                matrix[l] = np.array(list_strong[l])

            matrix = np.transpose(matrix)
            X_test.append(agg)
            Y_test.append(matrix)

    X_test = np.array(X_test)
    plt.plot(X_test[0])
    plt.show()
    ANE = ANE(X_test, output_strong_test_o)
    print("ANE:")
    print(ANE)


