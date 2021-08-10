from absl import app
from absl import flags
import numpy as np
import time
import sys
import tensorflow as tf 
import tensorflow_datasets as tfds 
sys.path.append('../')
from ai4AM import ai4AM
import models_lib
import metrics 
import objectives as obj
import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'simclr', 'Pretraining mode')
flags.DEFINE_string('dataset', 'Ai4amData', 'Dataset name')
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
flags.DEFINE_boolean('debug', False, 'Debug mode')

tf.compat.v1.logging.set_verbosity(20)
print(tf.compat.v1.logging.get_verbosity())


def pretrain(pretrain_ds, model, optimizer, epochs, lineareval_epochs=1, lineareval_task=None,
             eval_ds=None, ckpt_manager=None): 
    """
    Pretrain model with SimCLR framework and conduct linear evaluation on task if needed

    Args
    pretrain_ds: train dataset
    model: model
    optimizer: optimizer to use
    epochs: epochs to finetune for
    lineareval_epochs: linear_eval epochs
    lineareval_task: Dictionary with task specifications for linear evaluation
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    ckpt_manager: ckpt manager that manages model saves
    """

    print('==========PRETRAIN(NS)==========')
    # data preprocessing 
    ds = pretrain_ds.map(data_utils.experiment_preprocess)
    ds = ds.shuffle(1000) #??
    ds = ds.batch(FLAGS.pretrain_bs)

    # loss and metrics
    criterion = obj.contrastive_loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_contrast_acc = tf.keras.metrics.Accuracy(name='train_constrastive_accuracy')

    # metric for optimal checkpoint saving
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
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    filter(lambda gv: gv[0] is not None, zip(gradients, model.trainable_variables))
                )
            train_loss.update_state(loss)

        print('contrastive loss', train_loss.result())
        print('contrastive accuracy', train_contrast_acc.result())

        # Train linear head for linear_epochs and evaluate linear_eval performance 
        if lineareval_task:
            linear_eval(pretrain_ds, model, lineareval_task, epochs=lineareval_epochs, eval_ds=eval_ds)

        time_elapsed = (time.perf_counter() - time_start)
        print('time per epoch', time_elapsed)

        # Save new best model
        if ckpt_manager is not None and train_contrast_acc.result() > max_acc: 
            ckpt_manager.save()

    return model

def linear_eval(train_ds, model, task, epochs=10, eval_ds=None): 
    """
    Linear evaluate pretrained model for task

    Args
    train_ds: train dataset
    model: model
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    """

    print('==========LINEAR EVAL==========')

    # Filter out undesired examples with excluded_label
    ds = train_ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.map(data_utils.finetune_preprocess)
    ds = ds.shuffle(1000)
    ds = ds.batch(FLAGS.lineareval_bs)

    # loss, metrics, optimizers
    train_loss= tf.keras.metrics.Mean(name='train_loss')
    train_sup_acc = tf.keras.metrics.Accuracy(name='train_supervised_accuracy')
    criterion_sup = tf.nn.softmax_cross_entropy_with_logits 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Base network and projection head layers are not trainable in linear_eval
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

        # Evaluate results on eval_ds if possible
        if eval_ds is not None:
            evaluate(eval_ds, model, task)
    
    model.resnet.trainable = True
    model.ph.trainable = True

def control_vec_pretrain(pretrain_ds, model, optimizer, task, epochs=10, lineareval_epochs=1,
                         lineareval_task=None, eval_ds=None, ckpt_manager=None): 
    """
    Pretrain model with proposed control vector technique and conduct linear evaluation on task
    if needed. 

    Args
    pretrain_ds: pretraining dataset
    model: model
    optimizer: optimizer to use
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    lineareval_epochs: linear_eval epochs
    lineareval_task: Dictionary with task specifications for linear evaluation
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    """

    # Filter out undesired examples with excluded_label(s)
    ds = pretrain_ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.map(data_utils.control_vec_preprocess)
    ds = ds.shuffle(1000)
    ds = ds.batch(FLAGS.finetune_bs)

    # loss, metrics, optimizers
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_sup_acc = tf.keras.metrics.Accuracy(name='train_supervised_accuracy')
    criterion_sup = tf.nn.softmax_cross_entropy_with_logits 

    # metric for optimal checkpoint saving
    max_acc = 0.0

    for epoch in range(epochs): 
        train_loss.reset_states()
        train_sup_acc.reset_states()
        for x in ds:
            with tf.GradientTape() as tape:
                image = x['image']
                labels = x[task['name']]
                out = model(image, mode='supervised', sup_layers=2, training=True)
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

        if ckpt_manager is not None and train_sup_acc.result() > max_acc: 
            ckpt_manager.save()

        # Train linear head for linear_epochs and evaluate linear_eval performance 
        if lineareval_task:
            linear_eval(train_ds, model, lineareval_task, lineareval_epochs, eval_ds)

def finetune(ft_ds, model, task, epochs=10, eval_ds=None): 
    """
    Finetune model with labels for specified task. 

    Args
    ft_ds: dataset for finetuning
    model: model
    task: Dictionary with task specifications
    epochs: epochs to finetune for
    eval_ds: evaluation dataset, None if no evaluation needed after each epoch
    """

    print('==========FINETUNE==========')

    # Filter out undesired examples with excluded_label
    ds = ft_ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.map(data_utils.finetune_preprocess)
    ds = ds.shuffle(1000)
    ds = ds.batch(FLAGS.finetune_bs)

    # loss, metrics, optimizers
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
                # print(tf.math.argmax(out, axis=-1))
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

        # Evaluate results on eval_ds if possible
        if eval_ds: 
            evaluate(eval_ds, model, task)

def evaluate(eval_ds, model, task): 
    """
    Evaluate model on eval_ds for task. 

    Args
    eval_ds: dataset for evaluation
    model: model
    task: Dictionary with task specifications
    """

    print('==========EVAL==========')
    # Testing contrastive accuracy
    if task['name'] == 'contrastive_accuracy':
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
        return 

    # Testing classification accuracy 
    ds = eval_ds.filter(lambda x: x['label'] != task['excluded_label'])
    ds = ds.map(data_utils.eval_preprocess)
    ds = ds.batch(FLAGS.eval_bs)
    test_class_acc = tf.keras.metrics.Accuracy(name='test_class_accuracy')
    for x in ds:
        image = x['image']
        labels = x[task['name']]
        if task['name'] == 'extr':
            out = model(image, mode='eval', sup_layers=2, training=False)
        else:
            out = model(image, mode='eval', sup_layers=1, training=False)
        metrics.update_supervised_accuracy(test_class_acc, labels, out)
    
    if FLAGS.debug:
        print(tf.math.argmax(out, axis=-1))
    print('test classification accuracy')
    print(test_class_acc.result())

def load_model(path, model, optimizer): 
    """
    Load model and optimizer from path. 

    Args
    path: path/to/checkpoint
    model: model to load to
    optimizer: optimizer to load to
    """
    print("LOADING MODEL...")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(path))
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, 
        directory=FLAGS.model_dir, 
        max_to_keep=3 
    )
    return model, optimizer, ckpt, ckpt_manager

def train_with_control_vec_pretrianing(builder, train_ds, eval_ds):
    """
    Train with control vector pretraining and conduct linear evaluation

    Args
    builder: dataset builder
    train_ds: training set
    eval_ds: evaluation set
    """
    num_train_examples = builder.info.splits['train'].num_examples
    
    task = {'name': 'extr', 'excluded_label': None}
    num_classes = builder.info.features[task['name']].num_classes 
    task['num_classes'] = num_classes

    model = models_lib.Model(num_classes=num_classes)

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

    print('==========CONTROL VECTOR PRETRAIN==========')
    for epoch in range(FLAGS.pretrain_epochs):
        print('==========EPOCH: %s==========' % epoch)
        control_vec_pretrain(
            pretrain_ds=train_ds,
            model=model,
            optimizer=optimizer,
            task=task,
            epochs=1,
            lineareval_epochs=0,
            lineareval_task=None,
            eval_ds=None,
            ckpt_manager=ckpt_manager
        )

        head = model.sh
        model.sh = models_lib.SupervisedHead(main_task['num_classes'])
        linear_eval(train_ds, model, main_task, FLAGS.lineareval_epochs, eval_ds=eval_ds)

        model.sh = head

    model.sh = models_lib.SupervisedHead(main_task['num_classes'])
    linear_eval(train_ds, model, main_task, 30, eval_ds=eval_ds)
    evaluate(eval_ds, model, main_task)

def train_with_simclr_framework(builder, train_ds, eval_ds):
    """
    Train with simclr pretraining framework and conduct linear evaluation

    Args
    builder: dataset builder
    train_ds: training set
    eval_ds: evaluation set
    """
    num_train_examples = builder.info.splits['train'].num_examples
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
    model = models_lib.Model(num_classes=num_classes)

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

    pretrain(pretrain_ds=train_ds, model=model, optimizer=optimizer, epochs=FLAGS.pretrain_epochs, 
             lineareval_epochs=FLAGS.lineareval_epochs, lineareval_task=main_task, eval_ds=eval_ds, 
             ckpt_manager=ckpt_manager)

    linear_eval(train_ds, model, main_task, 30, eval_ds)
    evaluate(eval_ds, model, main_task)

def main(argv): 
    builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset()

    train_ds = dataset['train']
    train_ds = train_ds.map(data_utils.crop_bbox)

    eval_ds = dataset['validation'] 
    eval_ds = eval_ds.map(data_utils.crop_bbox)

    if FLAGS.mode == 'control_vector':
        train_with_control_vec_pretrianing(builder, train_ds, eval_ds)
    elif FLAGS.mode == 'simclr':
        train_with_simclr_framework(builder, train_ds, eval_ds)

    
if __name__ == '__main__':
    app.run(main)
