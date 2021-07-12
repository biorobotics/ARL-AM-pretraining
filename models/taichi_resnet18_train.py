import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pickle
import resnet

import taichiUtil

print('--------------------------------------Entering---------------------------------------')

batch_size = 20
vals_ds = tfds.load('taichiSim', split=[
    tfds.core.ReadInstruction('tmp', from_=k, to=k+10, unit='%')
    for k in range(0, 100, 10)], batch_size=batch_size)
trains_ds = tfds.load('taichiSim', split=[
    (tfds.core.ReadInstruction('tmp', to=k, unit='%') +
     tfds.core.ReadInstruction('tmp', from_=k+10, unit='%'))
    for k in range(0, 100, 10)], batch_size=batch_size)


model = resnet.resnet_18(drop_p=0, n_class=2)
objective = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

def loss(m, x, y, _train=True):
    _y = m(x, training=_train)
    return objective(y_true=y, y_pred=_y)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, _train=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

train_losses = []
train_accs = []
test_losses = []
test_accs = []

train = trains_ds[0]
val = vals_ds[0]

print('--------------------------------------Starting training!---------------------------------------')
n_epoch = 500
for epoch in range(n_epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_epoch_loss_avg = tf.keras.metrics.Mean()
    test_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train2 = train.shuffle(buffer_size=50)
    for i, ex in enumerate(train2):
        x = tf.image.resize(tf.image.convert_image_dtype(ex['image'], tf.float32), [250, 250])
        ys = taichiUtil.dat2vec(ex)
        # class_encode = tf.reshape(tf.one_hot((ys[:,3:7] > 0).astype(int), 2), (-1, 8))
        y = (ys[:,3] > 0).astype(int)
        l, grads = grad(model, x, y)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(l)
        epoch_accuracy.update_state(y, model(x, training=True))

        print(f'Train iter {i}/{len(train2)}', end='\r')

    confus = tf.zeros([2, 2], tf.int32)
    for ex in val:
        x = tf.image.resize(tf.image.convert_image_dtype(ex['image'], tf.float32), [250, 250])
        ys = taichiUtil.dat2vec(ex)
        y = (ys[:,3] > 0).astype(int)

        y_ = model(x, training=False)
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

