import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pickle
import resnet

import taichiUtil

class MyMo(tf.keras.Model):
    def __init__(self, im_head, n_class=4):
        super(MyMo, self).__init__()
        
        self.im_head = im_head
        self.fc1 = tf.keras.layers.Dense(units=n_class, activation=tf.keras.activations.softmax)
#         self.fc2 = tf.keras.layers.Dense(units=n_class, activation=tf.keras.activations.softmax)

#     def call(self, inputs, training=None):
    def call(self, im, uvec, training=None):
#         im, uvec = inputs
        x = self.im_head(im, training=training)
        x = tf.concat([x, uvec], axis=-1)
        out = self.fc1(x, training=training)
        return out

print('--------------------------------------Entering---------------------------------------')

batch_size = 20
# vals_ds = tfds.load('taichiSim', split=[
#     tfds.core.ReadInstruction('tmp', from_=k, to=k+10, unit='%')
#     for k in range(0, 100, 10)], batch_size=batch_size)
# trains_ds = tfds.load('taichiSim', split=[
#     (tfds.core.ReadInstruction('tmp', to=k, unit='%') +
#      tfds.core.ReadInstruction('tmp', from_=k+10, unit='%'))
#     for k in range(0, 100, 10)], batch_size=batch_size)

def preprocess(ex):
    ex['image'] = tf.image.resize(tf.image.convert_image_dtype(ex['image'], tf.float32), [250, 250])
    return ex
    # ex['image']

def pred(ex):
    return (ex['n_seq'] % 10 == 0)

train = tfds.load('taichiSim', split='train[10%:]').map(preprocess).filter(pred)
valid = tfds.load('taichiSim', split='train[:10%]').map(preprocess).filter(pred).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

resn = resnet.resnet_18(drop_p=0, n_class=2048, act=tf.keras.activations.relu)
model = MyMo(resn, n_class=2)
objective = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

def loss(m, x, y, _train=True):
    _y = m(*x, training=_train)
    return objective(y_true=y, y_pred=_y)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, _train=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

train_losses = []
train_accs = []
test_losses = []
test_accs = []

print('--------------------------------------Starting training!---------------------------------------')
n_epoch = 500
for epoch in range(n_epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_epoch_loss_avg = tf.keras.metrics.Mean()
    test_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train2 = train.shuffle(buffer_size=50).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    for i, ex in enumerate(train2):
        # x = tf.image.resize(tf.image.convert_image_dtype(ex['image'], tf.float32), [250, 250])
        x = ex['image']
        ys = taichiUtil.dat2vec(ex)
        # class_encode = tf.reshape(tf.one_hot((ys[:,3:7] > 0).astype(int), 2), (-1, 8))
        y = (ys[:,3] > 0).astype(int)
        l, grads = grad(model, (x, ys), y)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(l)
        epoch_accuracy.update_state(y, model(x, ys, training=True))

        print(f'Train iter {i}/128(?)', end='\r')

    confus = tf.zeros([2, 2], tf.int32)
    for ex in valid:
        # x = tf.image.resize(tf.image.convert_image_dtype(ex['image'], tf.float32), [250, 250])
        x = ex['image']
        ys = taichiUtil.dat2vec(ex)
        y = (ys[:,3] > 0).astype(int)

        y_ = model(x, ys, training=False)
        l = objective(y_true=y, y_pred=y_)
        test_epoch_loss_avg.update_state(l)
        test_epoch_accuracy.update_state(y, y_)

        confus += tf.math.confusion_matrix(y, tf.math.argmax(y_, axis=1))

    train_losses.append(epoch_loss_avg.result())
    train_accs.append(epoch_accuracy.result())
    test_losses.append(test_epoch_loss_avg.result())
    test_accs.append(test_epoch_accuracy.result())

    print(f'Epoch {epoch:03d}, Train Loss: {epoch_loss_avg.result():.3f}, Acc: {epoch_accuracy.result():.3f}')
    print(f'\tValidation Loss: {test_epoch_loss_avg.result():.3f}, Acc: {test_epoch_accuracy.result():.3f}')
    print(confus.numpy())