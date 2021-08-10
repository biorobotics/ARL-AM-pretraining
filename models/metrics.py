from absl import flags
import tensorflow as tf 
import numpy as np

FLAGS = flags.FLAGS

def update_contrastive_accuracy(metric, logits): 
    batch_size = logits.shape[0]
    gt_idx = tf.zeros(batch_size)
    y_idx = tf.math.argmax(logits, axis=-1)
    metric.update_state(gt_idx, y_idx)

def update_supervised_accuracy(metric, y_true, y_pred): 
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = tf.math.argmax(y_pred, axis=-1)
    metric.update_state(y_true, y_pred)