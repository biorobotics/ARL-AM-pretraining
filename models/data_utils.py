from absl import flags
import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np

FLAGS = flags.FLAGS

# Hyperparameters 

def random_crop_and_resize(image, strength=0.5): 
    aspect_ratio = 0.25 * strength 
    area = 0.95 * strength
    begin, size, bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image), 
        bounding_boxes=[[[0.0, 0.0, 1.0, 1.0]]],
        use_image_if_no_bounding_boxes=True, 
        aspect_ratio_range=[1.0-aspect_ratio, 1+aspect_ratio], 
        area_range=[1.0-area, 1.0]
    )
    image = tf.image.crop_and_resize([image], bbox[0], [0], [FLAGS.img_size, FLAGS.img_size])[0]
    return image

def random_greyscale(image): 
    p = 0.2
    image = tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), 
            tf.cast(p, tf.float32)
        ), 
        lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3]),
        lambda: image
    )
    return image

def random_jittering(image, strength=0.5): 
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    image = tf.clip_by_value(image, 0., 1.)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    image = tf.clip_by_value(image, 0., 1.)
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.clip_by_value(image, 0., 1.)
    image = tf.image.random_hue(image, max_delta=hue)
    image = tf.clip_by_value(image, 0., 1.)
    image = random_greyscale(image)
    return image

def random_jittering1(image, strength=0.5): 
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    image = tf.image.random_hue(image, max_delta=hue)
    image = tf.clip_by_value(image, 0., 1.)
    return image

def random_flip(image): 
    p = 0.2
    image = tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), 
            tf.cast(p, tf.float32)
        ), 
        lambda: tf.image.flip_left_right(image), 
        lambda: image
    )
    return image

def random_rotate(image): 
    max_angle = 15
    random_fn = tf.random.uniform # normal? 
    image = tfa.image.rotate(image, random_fn([], -max_angle, max_angle))
    return image

def random_cutout(image): 
    pass

def experiment_transforms(image): 
    # image = random_crop_and_resize(image)
    # image = random_rotate(image)
    # image = random_greyscale(image)
    image = random_jittering(image)
    return image

def pretrain_transforms(image): 
    image = random_crop_and_resize(image, 0.1)
    image = random_flip(image)
    image = random_jittering(image, 0.2)
    return image

def control_vec_transforms(image):
    # image = random_crop_and_resize(image, 0.1)
    # image = random_flip(image)
    # image = random_jittering(image, 0.2)
    return image

def finetune_transforms(image): 
    # image = random_crop_and_resize(image, 0.1)
    # image = random_flip(image)
    # image = random_jittering(image, 0.2)
    return image

def evaluate_transforms(image): 
    pass

def crop_bbox(x): 
    image = tf.image.crop_and_resize(
        image = [x['image']], 
        boxes = [x['bbox']], 
        box_indices = [0], 
        crop_size = [400, 400]
    )[0, :, :, :]
    x['image'] = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])
    return x

def experiment_preprocess(x): 
    image = x['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    x['image'] = [experiment_transforms(image), image]
    # x['image'] = [experiment_transforms(image), random_crop_and_resize(image)]
    return x

def control_vec_preprocess(x):
    image = x['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = control_vec_transforms(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])
    x['image'] = image
    return x

def pretrain_preprocess(x): 
    image = x['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    x['image'] = [pretrain_transforms(image), pretrain_transforms(image)]
    return x

def finetune_preprocess(x): 
    image = x['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = finetune_transforms(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])
    x['image'] = image
    return x

def eval_preprocess(x): 
    image = x['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = center_crop(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])
    x['image'] = image
    return x