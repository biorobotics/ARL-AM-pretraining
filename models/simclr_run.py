from absl import app
from absl import flags
import numpy as np
import time
import tensorflow as tf 
import tensorflow_datasets as tfds 

from dataset.AI4AM.ai4AM import ai4AM
import models 
import metrics 
import objectives as obj
import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/data', 'Data directory.')
flags.DEFINE_float('momentum', 0.99, 'Momentum.', lower_bound=0.0)
flags.DEFINE_integer('img_size', 200, 'Size of input image.', lower_bound=0)
flags.DEFINE_float('temp', 0.5, 'Temperature.', lower_bound=0.0)
flags.DEFINE_boolean('save_model', True, 'Whether to save model.')
flags.DEFINE_string('model_dir', './tmp_model', 'Directory to save model.')
flags.DEFINE_string('ckpt', None, 'Path to load checkpoint.')
flags.DEFINE_integer('pretrain_epochs', 50, 'Epochs for pretraining.')
flags.DEFINE_integer('finetune_epochs', 30, 'Epochs for finetuning.')
flags.DEFINE_integer('lineareval_epochs', 3, 'Epochs for linear eval.')
flags.DEFINE_integer('pretrain_bs', 32, 'Batch size for pretraining.')
flags.DEFINE_integer('finetune_bs', 128, 'Batch size for finetuning.')
flags.DEFINE_integer('lineareval_bs', 128, 'Batch size for linear evaluation.')
flags.DEFINE_integer('eval_bs', 128, 'Batch size for evaluation.')
flags.DEFINE_bool('debug', False, 'Print debug')

tf.compat.v1.logging.set_verbosity(20)
print(tf.compat.v1.logging.get_verbosity())
    
def pretrain(train_ds, model, optimizer, epochs, linear_epochs=1,
                     task=None, eval_ds=None, ckpt_manager=None): 
    """
    Pretrain model with SimCLR framework and conduct linear evaluation on task if needed

    Args
    train_ds: train dataset
    model: model
    optimizer: optimizer to use
    epochs: epochs to finetune for
    linear_epochs: linear_eval epochs
    task: Dictionary with task specifications
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    ckpt_manager: ckpt manager that manages model saves
    """
    print('==========PRETRAIN(NS)==========')
    # data preprocessing 
    ds = train_ds.map(data_utils.experiment_preprocess)
    ds = ds.shuffle(1000)
    ds = ds.batch(FLAGS.pretrain_bs)

    # loss and metrics
    criterion = obj.contrastive_loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_contrast_acc = tf.keras.metrics.Accuracy(name='train_constrastive_accuracy')
    max_acc = 0.0

    for epoch in range(epochs): 
        print('==========EPOCH: %s==========' % epoch)
        time_start = time.perf_counter()
        train_loss.reset_states()
        train_contrast_acc.reset_states()
        for x in ds:
            with tf.GradientTape() as tape:
                image = x['image']
                image = tf.transpose(image, [1, 0, 2, 3, 4])
                image = tf.reshape(
                    image, 
                    (image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4])
                )
                out = model(image, mode='unsupervised', training=True)
                loss, logits = criterion(None, out)
                metrics.update_contrastive_accuracy(train_contrast_acc, logits)
                if tf.math.is_nan(loss): 
                    print(out)
                    exit()
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    filter(lambda gv: gv[0] is not None, zip(gradients, model.trainable_variables))
                )
            train_loss.update_state(loss)
        print('contrastive loss', train_loss.result())
        print('contrastive accuracy', train_contrast_acc.result())

        # lineareval for this epoch
        if linear_epochs: 
            linear_eval(train_ds, model, task, epochs=linear_epochs, eval_ds=eval_ds)
        time_elapsed = (time.perf_counter() - time_start)
        print('time per epoch', time_elapsed)
        # save best model
        if ckpt_manager is not None and train_contrast_acc.result() > max_acc: 
            ckpt_manager.save()
    return model

def linear_eval(train_ds, model, task, epochs=10, eval_ds=None): 
    """
    Finetune pretrained model for task

    Args
    train_ds: train dataset
    model: model
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    """
    print('==========LINEAR EVAL==========')
    # ds = org_ds.filter(lambda x: x['label'] not in task['excluded_label'])
    ds = train_ds.filter(lambda x: x['label'] != 3)
    ds = ds.map(data_utils.finetune_preprocess)
    # ds = org_ds.map(data_utils.finetune_preprocess)
    ds = ds.shuffle(1000) #??
    ds = ds.batch(FLAGS.lineareval_bs)
    train_loss= tf.keras.metrics.Mean(name='train_loss')
    train_sup_acc = tf.keras.metrics.Accuracy(name='train_supervised_accuracy')
    criterion_sup = tf.nn.softmax_cross_entropy_with_logits 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.resnet.trainable = False
    model.ph.trainable = False 
    for epoch in range(epochs): 
        model.resnet.trainable = False
        train_loss.reset_states()
        train_sup_acc.reset_states()
        for x in ds:
            with tf.GradientTape() as tape:
                image = x['image']
                labels = x[task['name']]
                out = model(image, mode='supervised', sup_layers=1, training=True)
                # print(tf.math.argmax(out, axis=-1))
                metrics.update_supervised_accuracy(train_sup_acc, labels, out)
                loss = criterion_sup(tf.one_hot(labels, depth=task['num_classes']), out)
                loss = tf.math.reduce_mean(loss)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    filter(lambda gv: gv[0] is not None, zip(gradients, model.trainable_variables))
                )
            train_loss.update_state(loss)
        print('supervised (linear eval) loss')
        print(train_loss.result())
        print('supervised (linear eval) accuracy')
        print(train_sup_acc.result())
        if eval_ds is not None: 
            evaluate(eval_ds, model, task)
    model.resnet.trainable = True
    model.ph.trainable = True

def supervised_train(train_ds, model, task, epochs=10, eval_ds=None): 
    """
    Finetune pretrained model for task

    Args
    train_ds: train dataset
    model: model
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    """
    print('==========SUPERVISED TRAIN==========')

    # preprocess
    ds = train_ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.map(data_utils.finetune_preprocess)
    ds = ds.shuffle(1000)
    ds = ds.batch(FLAGS.finetune_bs)

    # setup
    train_loss= tf.keras.metrics.Mean(name='train_loss')
    train_sup_acc = tf.keras.metrics.Accuracy(name='train_supervised_accuracy')
    criterion_sup = tf.nn.softmax_cross_entropy_with_logits 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(epochs): 
        train_loss.reset_states()
        train_sup_acc.reset_states()
        for x in ds:
            with tf.GradientTape() as tape:
                image = x['image']
                labels = x[task['name']]
                out = model(image, mode='supervised', sup_layers=1, training=True)
                if FLAGS.debug: 
                    print(tf.math.argmax(out, axis=-1))
                metrics.update_supervised_accuracy(train_sup_acc, labels, out)
                loss = criterion_sup(tf.one_hot(labels, depth=task['num_classes']), out)
                loss = tf.math.reduce_mean(loss)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    filter(lambda gv: gv[0] is not None, zip(gradients, model.trainable_variables))
                )
            train_loss.update_state(loss)
        print('supervised train loss')
        print(train_loss.result())
        print('supervised train accuracy')
        print(train_sup_acc.result())
        if eval_ds: # if there is dataset to evaluate
            evaluate(eval_ds, model, task)

def finetune(ft_ds, model, task, epochs=10): 
    """
    Finetune pretrained model for task

    Args
    ft_ds: finetune dataset
    model: model
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    """
    print('==========FINETUNE==========')
    # preprocess
    ds = ft_ds.map(data_utils.finetune_preprocess)
    for l in task['excluded_label']: # exclude unlabeled data
        ds = ds.filter(lambda x: x['label'] != l)
    ds = ds.shuffle(1000) #??
    ds = ds.batch(FLAGS.finetune_bs)

    # training setup
    train_loss= tf.keras.metrics.Mean(name='train_loss')
    train_sup_acc = tf.keras.metrics.Accuracy(name='train_supervised_accuracy')
    criterion_sup = tf.nn.softmax_cross_entropy_with_logits 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

    # training loop
    for epoch in range(epochs): 
        train_loss.reset_states()
        train_sup_acc.reset_states()
        for x in ds:
            with tf.GradientTape() as tape:
                image = x['image']
                labels = x[task['name']]
                out = model(image, mode='supervised', sup_layers=1, training=True)
                metrics.update_supervised_accuracy(train_sup_acc, labels, out)
                loss = criterion_sup(tf.one_hot(labels, depth=task['num_classes']), out)
                loss = tf.math.reduce_mean(loss)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    filter(lambda gv: gv[0] is not None, zip(gradients, model.trainable_variables))
                )
            train_loss.update_state(loss)
        print('supervised loss')
        print(train_loss.result())
        print('supervised accuracy')
        print(train_sup_acc.result())

def eval_contrastive_accuracy(eval_ds_ds, model, task):
    ds = eval_ds.map(data_utils.pretrain_preprocess)
    ds = ds.batch(128)
    test_contrast_acc = tf.keras.metrics.Accuracy(name='test_constrastive_accuracy')
    for x in ds:
        image = x['image']
        image = tf.transpose(image, [1, 0, 2, 3, 4])
        image = tf.reshape(
            image, 
            (image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4])
        )
        out = model(image, mode='unsupervised', training=False)
        metrics.update_contrastive_accuracy2(test_contrast_acc, out, TEMP)
    print('test contrastive accuracy')
    print(test_contrast_acc.result())

def evaluate(eval_ds, model, task): 
    """
    Evaluate model performace for task on eval dataset

    Args
    eval_ds: evaluation dataset
    model: model
    task: Dictionary with task specifications
    """
    print('==========TEST==========')
    # Preprocess
    ds = eval_ds.map(data_utils.eval_preprocess)
    ds = ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.batch(FLAGS.eval_bs)
    test_class_acc = tf.keras.metrics.Accuracy(name='test_class_accuracy')

    for x in ds:
        image = x['image']
        labels = x[task['name']]
        out = model(image, mode='eval', sup_layers=1, training=False)

        if FLAGS.debug:
            print(model.training)
            print(tf.math.argmax(out, axis=-1))

        metrics.update_supervised_accuracy(test_class_acc, labels, out)
    print('test classification accuracy')
    print(test_class_acc.result())

def load_model(path, model, optimizer): 
    """
    Load latest saved model

    Args
    path: path to directory containing saved models
    model: object of model class to load to
    optimizer: object of optimizer class to load to
    """
    print("LOADING MODEL...")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(path))
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, 
        directory=FLAGS.model_dir, 
        max_to_keep=3 
        # ckpt_name??
    )
    return model, optimizer, ckpt, ckpt_manager

def main(argv): 
    builder = tfds.builder('Ai4amData', data_dir=FLAGS.data_dir)
    builder.download_and_prepare()
    num_train_examples = builder.info.splits['train'].num_examples

    dataset = builder.as_dataset()

    train_ds = dataset['train'] 
    train_ds = train_ds.map(data_utils.crop_bbox)

    eval_ds = dataset['validation'] 
    eval_ds = eval_ds.map(data_utils.crop_bbox)
    
    main_task = {'name': 'label', 'excluded_label': 3}
    num_classes = builder.info.features[main_task['name']].num_classes - 1
    main_task['num_classes'] = num_classes 

    lr_scheduler = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=0.0001, 
        first_decay_steps=10*(num_train_examples//FLAGS.pretrain_bs),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    model = models.Model(num_classes=num_classes)

    if FLAGS.ckpt: 
        model, optimizer, ckpt, ckpt_manager = load_model(FLAGS.ckpt, model, optimizer)
    else: 
        if FLAGS.save_model: 
            ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
            ckpt_manager = tf.train.CheckpointManager(
                checkpoint=ckpt, 
                directory=FLAGS.model_dir+'/pretrain', 
                max_to_keep=3 
            )
        else: 
            ckpt=None
            ckpt_manager=None

    # pretrain(train_ds=train_ds, model=model, optimizer=optimizer, epochs=FLAGS.pretrain_epochs, 
    #          linear_epochs=FLAGS.lineareval_epochs, task=main_task, eval_ds=eval_ds, 
    #          ckpt_manager=ckpt_manager)
    # model = train_no_shuffle(train_ds, model, optimizer, 1, ckpt, ckpt_manager)

    linear_eval(train_ds, model, main_task, 100, eval_ds)
    evaluate(eval_ds, model, main_task)

    # model, _, _ = load_model('./test_model/ckpt-100.index')
    exit()
if __name__ == '__main__':
    app.run(main)
