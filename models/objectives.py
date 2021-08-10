from absl import flags
import tensorflow as tf 
import numpy as np

FLAGS = flags.FLAGS

def contrastive_loss(y_true, y_pred, temperature=None): 
    if temperature == None: 
        temperature = FLAGS.temp
    batch_size = y_pred.shape[0]
    y_norm = tf.math.divide(y_pred, tf.math.reduce_euclidean_norm(y_pred, axis=-1, keepdims=True))
    y_sim = tf.repeat(tf.expand_dims(y_norm, axis=1), repeats=batch_size, axis=1)
    y_sim = tf.einsum('ijk,jk->ij', y_sim, y_norm)
    mask = 1 - tf.eye(batch_size) 
    labels = tf.concat([tf.range(batch_size//2), tf.range(batch_size//2)], axis=-1)
    labels = (tf.expand_dims(labels, axis=0) == tf.expand_dims(labels, axis=1))
    negatives = tf.reshape(y_sim[tf.math.logical_not(labels)], shape=(batch_size, batch_size-2))
    mask = tf.cast(tf.eye(batch_size, dtype=tf.int8), dtype=tf.bool)
    positives = y_sim[tf.math.logical_xor(labels, mask)]
    logits = tf.concat([tf.expand_dims(positives, axis=-1), negatives], axis=-1)
    logits = tf.nn.softmax(logits / temperature, axis=-1)
    loss = logits[:, 0]
    loss = -1*tf.math.log(loss)
    return tf.math.reduce_mean(loss), logits