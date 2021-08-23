# Lint as: python2, python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training script for a semantic segmentation model base on U-Net

See model.py for more details and usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import model
from datasets import data_generator
from utils import utils
import matplotlib.pyplot as plt

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

# Momentum optimizer flags

flags.DEFINE_enum('optimizer', 'momentum', ['momentum', 'adam', "RMSprop"],
                  'Which optimizer to use.')

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('decay_steps', 0.0,
                   'Decay steps for polynomial learning rate schedule.')

flags.DEFINE_float('end_learning_rate', 0.0,
                   'End learning rate for polynomial learning rate schedule.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# Adam optimizer flags
flags.DEFINE_float('adam_learning_rate', 0.001,
                   'Learning rate for the adam optimizer.')
flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')

flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_list('train_crop_size', '513,513',
                  'Image crop size [height, width] during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('checkpoint_path', "./checkpoints/",
                    'The checkpoint_path.')

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')
# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def main(unused_argv):
    tf.io.gfile.makedirs(FLAGS.train_logdir)
    tf.compat.v1.logging.info('Training on %s set', FLAGS.train_split)

    dataset_training = data_generator.Dataset(
          dataset_name=FLAGS.dataset,
          split_name=FLAGS.train_split,
          dataset_dir=FLAGS.dataset_dir,
          batch_size=FLAGS.train_batch_size,
          crop_size=[int(sz) for sz in FLAGS.train_crop_size],
          #min_resize_value=FLAGS.min_resize_value,
          #max_resize_value=FLAGS.max_resize_value,
          #resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          num_readers=4,
          is_training=True,
          should_shuffle=True,
          should_repeat=True)

    dataset_validation = data_generator.Dataset(
          dataset_name=FLAGS.dataset,
          split_name="val_fine",
          dataset_dir=FLAGS.dataset_dir,
          batch_size=FLAGS.train_batch_size,
          crop_size=[int(sz) for sz in FLAGS.train_crop_size],
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          scale_factor_step_size=FLAGS.scale_factor_step_size,
          num_readers=4,
          is_training=False,
          should_shuffle=True,
          should_repeat=True)

    num_classes = dataset_training.num_of_classes
    ignore_label = dataset_training.ignore_label

    train = dataset_training.get_dataset()
    val = dataset_validation.get_dataset()


    print("- Dataset num of classes: ", dataset_training.num_of_classes)
    print("- label to ignore ", dataset_training.ignore_label)
    print("- Train dataset: ", train)
    print("- Validation dataset : ", train)

    input_shape = [int(sz) for sz in FLAGS.train_crop_size]
    input_shape.append(3)
    # Define the model.
    model_fn = model.unet_model(input_shape=input_shape, n_classes=num_classes)

    # Build the optimizer
    learning_rate = utils.get_model_learning_rate(
          FLAGS.learning_policy,
          FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step,
          FLAGS.learning_rate_decay_factor,
          FLAGS.training_number_of_steps,
          FLAGS.learning_power,
          FLAGS.slow_start_step,
          decay_steps=FLAGS.decay_steps,
          end_learning_rate=FLAGS.end_learning_rate)

    if FLAGS.optimizer == 'momentum':
        optimizer = keras.optimizers.SGD(learning_rate, FLAGS.momentum)
    elif FLAGS.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=FLAGS.adam_learning_rate, epsilon=FLAGS.adam_epsilon)
    elif FLAGS.optimizer == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=0.001)
    else:
        raise ValueError('Unknown optimizer')

    model_fn.compile(loss=model.keras_custom_loss_function,
                     optimizer=optimizer,
                     metrics=['accuracy'])

    if  FLAGS.tf_initial_checkpoint:
        print("initial checkpoint")

    # Create a callback that saves the model's weights every 10 epochs
    checkpoint = tf.train.Checkpoint(model_fn)
    save_period = 10
    steps_per_epoch = 3475 // FLAGS.train_batch_size
    validation_steps = 500 // FLAGS.train_batch_size

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=int(save_period * steps_per_epoch))

    hist = model_fn.fit(train,
                        validation_data=val,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=30,
                        callbacks=[cp_callback])

    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model_fn.save("my_model")

    history = hist.history
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    plt.plot(acc, '-', label='Training Accuracy')
    plt.plot(val_acc, '--', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # It can be used to reconstruct the model identically.
    # model_fn = keras.models.load_model("my_model")

    # Restore the checkpointed values to the `model` object.
    # save_path = checkpoint.save(FLAGS.checkpoint_path)
    # checkpoint.restore(save_path)


if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.compat.v1.app.run()
