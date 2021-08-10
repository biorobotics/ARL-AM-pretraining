from absl import flags
import tensorflow as tf 
import numpy as np

FLAGS = flags.FLAGS

class ResidualBlock(tf.keras.layers.Layer): 
    def __init__(self, in_ch, dim_increase=False, dropout=None, name='residual_block'): 
        super(ResidualBlock, self).__init__(name=name)
        self.dim_increase = dim_increase
        if dim_increase: 
            out_ch = 2*in_ch
            self.conv1 = tf.keras.layers.Conv2D(out_ch, 3, strides=2, padding='same')
        else: 
            out_ch = in_ch 
            self.conv1 = tf.keras.layers.Conv2D(out_ch, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(out_ch, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
        self.relu2 = tf.keras.layers.ReLU()
    def call(self, inputs): 
        if self.dim_increase: 
            identity = tf.keras.layers.AveragePooling2D(3, strides=2, padding='same')(inputs)
            identity = tf.concat([identity, tf.zeros(identity.shape, identity.dtype)], axis=-1) # option A (should try option B)
        else: 
            identity = tf.identity(inputs)
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out) # batch norm??
        out = self.relu2(out+identity)
        return out

class ResNet(tf.keras.layers.Layer): 
    def __init__(self, architecture, width=1, block_type='residual', name='resnet'): 
        super(ResNet, self).__init__(name=name)
        channels = 64 * width
        self.conv1 = tf.keras.layers.Conv2D(channels, 7, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum) #confirm parameter
        self.relu1 = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        if block_type == 'residual': 
            block_fn = ResidualBlock

        blockgroup1 = []
        for b in range(architecture[0]): # 64 channels (type 1 blocks)
            blockgroup1.append(block_fn(channels))

        blockgroup2 = []
        blockgroup2.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[1]): # 128 channels
            blockgroup2.append(block_fn(channels))

        blockgroup3 = []
        blockgroup3.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[2]): # 256 channels 
            blockgroup3.append(block_fn(channels))

        blockgroup4 = []
        blockgroup4.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[3]): # 512 channels 
            blockgroup4.append(block_fn(channels))

        self.blocks = [blockgroup1, blockgroup2, blockgroup3, blockgroup4]
        self.global_avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs): 
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        for block_group in self.blocks: 
            for b in block_group: 
                out = b(out)

        out = self.global_avgpool(out)
        return out

class CifarResNet(tf.keras.layers.Layer): 
    def __init__(self, architecture, width=1, block_type='residual', name='resnet'): 
        super(CifarResNet, self).__init__(name=name)
        channels = 64 * width
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum) #confirm parameter
        self.relu1 = tf.keras.layers.ReLU()
        # self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        if block_type == 'residual': 
            block_fn = ResidualBlock

        blockgroup1 = []
        for b in range(architecture[0]): # 64 channels (type 1 blocks)
            blockgroup1.append(block_fn(channels))

        blockgroup2 = []
        blockgroup2.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[1]): # 128 channels
            blockgroup2.append(block_fn(channels))

        blockgroup3 = []
        blockgroup3.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[2]): # 256 channels 
            blockgroup3.append(block_fn(channels))

        blockgroup4 = []
        blockgroup4.append(block_fn(channels, dim_increase=True))
        channels *= 2
        for b in range(1, architecture[3]): # 512 channels 
            blockgroup4.append(block_fn(channels))

        self.blocks = [blockgroup1, blockgroup2, blockgroup3, blockgroup4]
        self.global_avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs): 
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        for block_group in self.blocks: 
            for b in block_group: 
                out = b(out)

        out = self.global_avgpool(out)
        return out

class ProjectionHead1(tf.keras.layers.Layer): 
    def __init__(self, proj_layers=2, proj_size=256, num_classes=0, name='proj_head'): # try layer and size
        super(ProjectionHead, self).__init__(name=name)
        self.proj_layers = proj_layers
        if proj_layers == 1: 
            self.fc1 = tf.keras.layers.Dense(proj_size, use_bias=bool(num_classes))
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
        if proj_layers == 2: 
            self.fc1 = tf.keras.layers.Dense(proj_size)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu1 = tf.keras.layers.ReLU()
            self.fc2 = tf.keras.layers.Dense(proj_size, use_bias=bool(num_classes))
            self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
        if proj_layers == 3: 
            self.fc1 = tf.keras.layers.Dense(proj_size)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu1 = tf.keras.layers.ReLU()
            self.fc2 = tf.keras.layers.Dense(proj_size)
            self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu2 = tf.keras.layers.ReLU()
            self.fc3 = tf.keras.layers.Dense(proj_size, use_bias=bool(num_classes))
            self.bn3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
        if proj_layers > 3: 
            raise Exception
        self.output_layer = None
        if num_classes:
            self.relu3 = tf.keras.layers.ReLU()
            self.output_layer = tf.keras.layers.Dense(num_classes)
    def call(self, inputs, mode='unsupervised', sup_layers=1): 
        assert(sup_layers <= self.proj_layers)
        if mode == 'unsupervised': 
            out = inputs
            if self.proj_layers >= 1: 
                out = self.fc1(inputs)
                out = self.bn1(out) 
            if self.proj_layers >= 2: 
                out = self.relu1(out) 
                out = self.fc2(out)
                out = self.bn2(out) 
            if self.proj_layers >= 3: 
                out = self.relu2(out) 
                out = self.fc3(out)
                out = self.bn3(out) 
                # out = self.relu3(out) # relu? 
            if self.output_layer:
                out = self.relu3(out)
                out = self.output_layer(out)
        else:  # supervised training and evaluation 
            out = inputs
            if sup_layers >= 1: 
                out = self.fc1(inputs)
                out = self.bn1(out) 
            if sup_layers >= 2: 
                out = self.relu1(out) 
                out = self.fc2(out)
                out = self.bn2(out) 
            if sup_layers >= 3: 
                out = self.relu2(out) 
                out = self.fc3(out)
                out = self.bn3(out) 
        return out

class ProjectionHead(tf.keras.layers.Layer): 
    def __init__(self, proj_layers=2, proj_size=256, num_classes=0, name='proj_head'): # try layer and size
        super(ProjectionHead, self).__init__(name=name)
        self.proj_layers = proj_layers
        if proj_layers >= 1: 
            self.fc1 = tf.keras.layers.Dense(proj_size, use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu1 = tf.keras.layers.ReLU()
        if proj_layers >= 2: 
            self.fc1 = tf.keras.layers.Dense(proj_size)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu1 = tf.keras.layers.ReLU()
            self.fc2 = tf.keras.layers.Dense(proj_size, use_bias=False)
            self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu2 = tf.keras.layers.ReLU()
        if proj_layers >= 3: 
            self.fc1 = tf.keras.layers.Dense(proj_size)
            self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu1 = tf.keras.layers.ReLU()
            self.fc2 = tf.keras.layers.Dense(proj_size)
            self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu2 = tf.keras.layers.ReLU()
            self.fc3 = tf.keras.layers.Dense(proj_size, use_bias=False)
            self.bn3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=FLAGS.momentum)
            self.relu3 = tf.keras.layers.ReLU()
        if proj_layers > 3: 
            raise Exception
        if num_classes:
            self.output_layer = tf.keras.layers.Dense(num_classes, use_bias=False)

    def call(self, inputs, mode='unsupervised', sup_layers=1): 
        assert(sup_layers <= self.proj_layers)
        if mode == 'unsupervised': 
            out = inputs
            if self.proj_layers >= 1: 
                out = self.fc1(inputs)
                out = self.bn1(out) 
            if self.proj_layers >= 2: 
                out = self.relu1(out) 
                out = self.fc2(out)
                out = self.bn2(out) 
            if self.proj_layers >= 3: 
                out = self.relu2(out) 
                out = self.fc3(out)
                out = self.bn3(out) 
                # out = self.relu3(out) # relu? 
        else:  # supervised training and evaluation 
            out = inputs
            if sup_layers >= 1: 
                out = self.fc1(inputs)
                out = self.bn1(out) 
            if sup_layers >= 2: 
                out = self.relu1(out) 
                out = self.fc2(out)
                out = self.bn2(out) 
            if sup_layers >= 3: 
                out = self.relu2(out) 
                out = self.fc3(out)
                out = self.bn3(out) 
        return out

class SupervisedHead(tf.keras.layers.Layer): 
    def __init__(self, num_classes, name='supervised_head'): 
        super(SupervisedHead, self).__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(num_classes)
    def call(self, inputs): 
        out = self.fc1(inputs)
        return out 

class Model(tf.keras.Model): 
    def __init__(self, proj_layers=2, proj_size=256, num_classes=2, name='model'): 
        super(Model, self).__init__(name=name)
        self.resnet = ResNet([3, 4, 6, 3])
        self.ph = ProjectionHead(proj_layers=proj_layers, proj_size=proj_size)
        self.sh = SupervisedHead(num_classes)
    def call(self, inputs, mode='unsupervised', sup_layers=None): 
        if mode == 'unsupervised': 
            h = self.resnet(inputs)
            z = self.ph(h)
            return z
        if mode == 'supervised': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='supervised', sup_layers=sup_layers)
            # print(sup_in)
            out = self.sh(sup_in)
            return out
        if mode == 'eval': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='eval', sup_layers=sup_layers)
            # print(sup_in)
            out = self.sh(sup_in)
            return out

class MultiheadModel(tf.keras.Model): 
    def __init__(self, tasks, proj_layers=2, proj_size=256, num_classes=2, name='model'): 
        super(Model, self).__init__(name=name)
        self.resnet = ResNet([3, 4, 6, 3])
        self.ph = ProjectionHead(proj_layers=proj_layers, proj_size=proj_size)
        self.sh = SupervisedHead(num_classes)
        task_heads = {}
        for k in tasks.keys():
            ph[tasks[k]['name']] = SupervisedHead(num_classes=tasks[k]['num_classes'])
        self.task_heads = task_heads
    def call(self, inputs, task_name=None, mode='unsupervised', sup_layers=None): 
        if mode == 'unsupervised': 
            h = self.resnet(inputs)
            z = self.ph(h)
            return z
        if mode == 'supervised': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='supervised', sup_layers=sup_layers)
            # print(sup_in)
            out = self.task_heads[task_name](sup_in)
            # out = self.sh(sup_in)
            return out
        if mode == 'eval': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='eval', sup_layers=sup_layers)
            # print(sup_in)
            out = self.sh(sup_in)
            return out

class CifarModel(tf.keras.layers.Layer): 
    def __init__(self, proj_layers=2, proj_size=256, num_classes=2, name='model'): 
        super(CifarModel, self).__init__(name=name)
        self.resnet = CifarResNet([2, 2, 2, 2])
        self.ph = ProjectionHead(proj_layers=proj_layers, proj_size=proj_size)
        self.sh = SupervisedHead(num_classes)
    def call(self, inputs, mode='unsupervised', sup_layers=None): 
        if mode == 'unsupervised': 
            h = self.resnet(inputs)
            z = self.ph(h)
            return z
        if mode == 'supervised': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='supervised', sup_layers=sup_layers)
            out = self.sh(sup_in)
            return out
        if mode == 'eval': 
            h = self.resnet(inputs)
            sup_in = self.ph(h, mode='eval', sup_layers=sup_layers)
            out = self.sh(sup_in)
            return out

# code below is directly from SimCLR official implementation
# https://github.com/google-research/simclr/blob/master/tf2/model.py
class WarmUpAndCosineDecayOrig(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self, base_learning_rate, num_examples, epochs, name=None):
    super(WarmUpAndCosineDecay, self).__init__()
    self.base_learning_rate = base_learning_rate
    self.num_examples = num_examples
    self.epochs = epochs
    self._name = name

  def __call__(self, step):
    with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
      warmup_steps = int(
          round(WARMUP_EPOCHS * self.num_examples //
                FLAGS.train_bs))
      scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_bs)
      learning_rate = (
          step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

      # Cosine decay learning rate schedule
      total_steps = self.num_examples * self.epochs
      # TODO(srbs): Cache this object.
      cosine_decay = tf.keras.experimental.CosineDecay(
          scaled_lr, total_steps - warmup_steps)
      learning_rate = tf.where(step < warmup_steps, learning_rate,
                               cosine_decay(step - warmup_steps))

      return learning_rate
