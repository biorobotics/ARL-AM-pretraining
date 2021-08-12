"""ai4am dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import re

import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# TODO(ai4am): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

3 fields: image, bbox, label. image is uncropped.
"""

# TODO(ai4am): BibTeX citation
_CITATION = """
"""


class AMPlasticDefects(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ai4AM dataset."""

    VERSION = tfds.core.Version('0.5.0')
    RELEASE_NOTES = {
        '0.2.0': 'Train, valid, test splits. Also a train_labeled which is a subset of train.',
        '0.3.0': 'Extrusion rate (extr) label added',
        '0.4.0': 'Layer number added, and unlabeled full images',
        '0.5.0': 'Change unlabeled->nobbox, add labels to that set.'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(ai4AM): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(720, 1280, 3)),
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(names=['Over-extrusion', 'Under-extrusion', 'Normal', 'Unlabeled']),
                'extr': tfds.features.ClassLabel(
                    names=['extr0.6', 'extr0.7', 'extr0.8', 'extr0.9', 'extr1', 'extr1.1', 'extr1.2']),
                'layer': tf.int32,
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(ai4AM): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            'https://drive.google.com/uc?id=1KTA_BAdh86Oo3RPrCMeo-zu2o2f_zCWV&export=download')
        # https://drive.google.com/file/d/1KTA_BAdh86Oo3RPrCMeo-zu2o2f_zCWV/view?usp=sharing

        # TODO(ai4AM): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(os.path.join(path, 'label_train.csv')),
            'train_nobbox': self._generate_examples(os.path.join(path, 'no_crop_train.csv')),
            'labeled_train': self._generate_examples(os.path.join(path, 'label_train_labeled.csv')),
            'validation': self._generate_examples(os.path.join(path, 'label_valid.csv')),
            'test': self._generate_examples(os.path.join(path, 'label_test.csv')),
        }

    def _generate_examples(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        prefix = path[:path.rindex('/') + 1]

        splitlines = map(lambda s: s.strip().split(','), lines)
        for i, l in enumerate(splitlines):
            fn, x0, y0, lab = l
            x0 = int(x0)
            y0 = int(y0)
            bbox_coords = x0 / 720, y0 / 1280, (x0 + 400) / 720, (y0 + 400) / 1280
            # im = cv2.imread(prefix+fn)

            extr = re.search(r'extr\d(\.\d)?', fn)[0]

            dr, zz = os.path.split(fn)
            files = sorted([s for s in os.listdir(os.path.join(prefix, dr)) if s.endswith('.png')])
            layer = files.index(zz)

            record = {
                'image': os.path.join(prefix, fn),
                'bbox': tfds.features.BBox(*bbox_coords),
                'label': lab,
                'extr': extr,
                'layer': layer
            }

            yield f'im{i:04d}.png', record
