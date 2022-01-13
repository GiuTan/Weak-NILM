import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras import backend as K


def binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true, new_pred)
    return tf.reduce_mean(loss)


def binary_crossentropy_weak(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    new_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))
    new_pred = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))

    loss = K.binary_crossentropy(new_true, new_pred)

    return tf.reduce_mean(loss)