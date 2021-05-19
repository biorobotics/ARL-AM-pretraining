import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pickle
import resnet

print('--------------------------------------Entering---------------------------------------')

batch_size = 10

# ds = tfds.load('ai4am_data', batch_size=batch_size)
# train = tfds.load('ai4am_data', batch_size=10, split='labeled_train')
train = tfds.load('ai4am_data', batch_size=10, split='train_nobbox')
# test = tfds.load('ai4am_data', batch_size=10, split='test')
valid = tfds.load('ai4am_data', batch_size=10, split='validation')

model_save = '/home/code/resnet18_randomcrops'
model_top_save = '/home/code/resnet18_randomcrops_top'

# model = tf.keras.applications.resnet.ResNet50(input_shape=(400, 400, 3), include_top=True, weights=None, classes=3)
model = resnet.resnet_18(drop_p=0)
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

print('--------------------------------------Starting training!---------------------------------------')

top_valid = 100

n_epoch = 500
for epoch in range(n_epoch):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  test_epoch_loss_avg = tf.keras.metrics.Mean()
  test_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  train2 = train.shuffle(buffer_size=50)
  for val in train2:
    bboxes = np.random.random(val['bbox'].shape)
    _, h, w, _ = val['image'].shape
    bboxes = bboxes * [(h-400)/h, (w-400)/w, 1, 1]
    bboxes[:, 2:] = bboxes[:, :2] + [400/h, 400/w]
    # print(bboxes)
    # exit(0)

    x = tf.image.crop_and_resize(val['image'], bboxes, range(len(val['bbox'])), (400, 400))
    y = val['label']
    l, grads = grad(model, x, y)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    epoch_loss_avg.update_state(l)
    epoch_accuracy.update_state(y, model(x, training=True))

  confus = tf.zeros([3, 3], tf.int32)
  for val in valid:
    x = tf.image.crop_and_resize(val['image'], val['bbox'], range(len(val['bbox'])), (400, 400))
    y = val['label']

    y_ = model(x, training=False)
    l = objective(y_true=y, y_pred=y_)
    test_epoch_loss_avg.update_state(l)
    test_epoch_accuracy.update_state(y, y_)

    confus += tf.math.confusion_matrix(val['label'], tf.math.argmax(y_, axis=1))

  train_losses.append(epoch_loss_avg.result())
  train_accs.append(epoch_accuracy.result())
  test_losses.append(test_epoch_loss_avg.result())
  test_accs.append(test_epoch_accuracy.result())

  print(f'Epoch {epoch:03d}, Train Loss: {epoch_loss_avg.result():.3f}, Acc: {epoch_accuracy.result():.3f}')
  print(f'\tValidation Loss: {test_epoch_loss_avg.result():.3f}, Acc: {test_epoch_accuracy.result():.3f}')
  print(confus.numpy())

  if test_epoch_loss_avg.result() < top_valid:
    top_valid = test_epoch_loss_avg.result()
    model.save(model_top_save)

model.save(model_save)
