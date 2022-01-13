import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# The energy error is the difference between
# the total predicted energy, and the total actual energy consumed
# by each active appliance in that sample instant


def ANE(X_test_synth,Y_pred):

    classes = 5
    fridge_val = 91
    kettle_val = 1996
    micro_val = 1107
    wash_val = 487
    dish_val = 723
    mean_val = [kettle_val,micro_val,fridge_val,wash_val,dish_val]
    p_actual = 0
    p_ave = 0
    for i in range(len(X_test_synth)):
        agg = X_test_synth[i]
        y_ave = Y_pred[i]
        y_ave_t = y_ave.transpose()
        p_ave_curr = 0
        for k in range(classes):
            y_ave_t[k] = y_ave_t[k] * mean_val[k]
            p_ave_curr = p_ave_curr + y_ave_t[k]
        p_actual = p_actual + agg
        p_ave = p_ave + p_ave_curr
    abs_ = np.abs(np.sum(p_actual) - np.sum(p_ave))
    print("True energy", np.sum(p_actual))
    print("Predicted energy", np.sum(p_ave))
    ANE = abs_ / np.sum(p_actual)

    return ANE

def custom_f1_score(y_true, y_pred):
    y_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    y_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    def recall_m(y_true, y_pred):
         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
         Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

         recall = TP / (Positives + K.epsilon())
         return recall

    def precision_m(y_true, y_pred):
         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
         Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

         precision = TP / (Pred_Positives + K.epsilon())
         return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
