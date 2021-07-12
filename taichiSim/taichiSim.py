"""taichi fluid sim dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import re
import yaml

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

class TaichiSim(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
        '0.0.0': 'Initial dataset',
        '0.1.0': 'Working initial. With download.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(720, 1280, 3)),
                # 'label': tfds.features.ClassLabel(names=['Over-extrusion', 'Under-extrusion', 'Normal', 'Unlabeled']),
                'fl_hardening': tf.float32,
                'fl_lambda0': tf.float32,
                'fl_mu0': tf.float32,
                'fl_thetac': tf.float32,
                'fl_thetas': tf.float32,
                'fl_type': tfds.features.ClassLabel(names=['snow', 'jelly', 'water', 'sand']),

                'm_x0': tf.float32,
                'm_y0': tf.float32,
                'm_dx': tf.float32,
                'm_dy': tf.float32,

                'ex_scale': tf.float32,
                'ex_v0': tf.float32,

                't_pause': tf.float32,
                't_seg': tf.float32,
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            # supervised_keys=('image', 'label'),  # e.g. ('image', 'label')
            # homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(ai4AM): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            'https://drive.google.com/uc?id=191Qknp6zuqYBrwb1ie0onur2wUA-6Qza&export=download')
        # https://drive.google.com/file/d/191Qknp6zuqYBrwb1ie0onur2wUA-6Qza/view?usp=sharing

        # TODO(ai4AM): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'tmp': self._generate_examples(path),
            # 'test': self._generate_examples(os.path.join(path, 'label_test.csv')),
        }

    def _generate_examples(self, path):
        path = os.path.join(path, 'taichi_sim_data')
        dirs = os.listdir(path)
        p = re.compile('render_t[0-9]+_flip')
        i = 0

        for d in dirs:
            if p.match(d) is None: continue
            fn = os.path.join(path, d)

            images = []
            config = None
            for file in os.listdir(fn):
                if file.endswith('.png'):
                    images.append(file)
                elif file.endswith('.yaml'):
                    config = file
            
            with open(os.path.join(fn, config), 'r') as f:
                parms = yaml.load(f)
            record = {
                'fl_hardening': parms['fluid'].get('hardening', 10),
                'fl_lambda0': parms['fluid'].get('lambda_0', 38888.9),
                'fl_mu0': parms['fluid'].get('mu_0', 58333.3),
                'fl_thetac': parms['fluid'].get('theta_c', 2.5e-2),
                'fl_thetas': parms['fluid'].get('theta_s', 7.5e-3),
                'fl_type': parms['fluid']['type'],
                'm_x0': parms['motion']['x0'],
                'm_y0': parms['motion']['y0'],
                'm_dx': parms['motion']['dx'],
                'm_dy': parms['motion']['dy'],
                'ex_scale': parms['fluid_scale'],
                'ex_v0': parms['v0'],
                't_pause': parms['pause_len'],
                't_seg': parms['seg_len'],
            }

            for im in images:
                seqnum = int(im[3:-4])
                if seqnum < 30: continue

                rec = record.copy()
                rec['image'] = os.path.join(fn, im)
                yield f'im{i:04d}.png', rec
                i += 1
                