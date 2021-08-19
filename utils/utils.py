# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Utility functions for training."""

import tensorflow as tf


def resize_image(image):
    image = tf.cast(image, tf.float32)
    # scale values to [0,1]
    image = image / 255.0
    # resize image
    image = tf.image.resize(image, (128, 128))
    return image


def resize_mask(mask):
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return mask


def brightness(img, mask):
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask


def gamma(img, mask):
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask


def hue(img, mask):
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask


def crop(img, mask):
    img = tf.image.central_crop(img, 0.7)
    img = tf.image.resize(img, (128, 128))
    mask = tf.image.central_crop(mask, 0.7)
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return img, mask


def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask


def flip_vert(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask


def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            decay_steps=0.0,
                            end_learning_rate=0.0
                            ):
      """Gets model's learning rate.

      Computes the model's learning rate for different learning policy.
      Right now, only "step" and "poly" are supported.
      (1) The learning policy for "step" is computed as follows:
        current_learning_rate = base_learning_rate *
          learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
      See tf.train.exponential_decay for details.
      (2) The learning policy for "poly" is computed as follows:
        current_learning_rate = base_learning_rate *
          (1 - global_step / training_number_of_steps) ^ learning_power

      Args:
        learning_policy: Learning rate policy for training.
        base_learning_rate: The base learning rate for model training.
        learning_rate_decay_step: Decay the base learning rate at a fixed step.
        learning_rate_decay_factor: The rate to decay the base learning rate.
        training_number_of_steps: Number of steps for training.
        learning_power: Power used for 'poly' learning policy.
        slow_start_step: Training model with small learning rate for the first
          few steps.
        decay_steps: Float, `decay_steps` for polynomial learning rate.
        end_learning_rate: Float, `end_learning_rate` for polynomial learning rate.

      Returns:
        Learning rate for the specified learning policy.

      """
      global_step = tf.compat.v1.train.get_or_create_global_step()
      # tf.train.get_or_create_global_step()
      adjusted_global_step = tf.maximum(global_step - slow_start_step, 0)
      if decay_steps == 0.0:
        tf.compat.v1.logging.info('Setting decay_steps to total training steps.')
        decay_steps = training_number_of_steps - slow_start_step
      if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            adjusted_global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
      elif learning_policy == 'poly':
        learning_rate = tf.compat.v1.train.polynomial_decay(
            base_learning_rate,
            adjusted_global_step,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=learning_power)
      else:
        raise ValueError('Unknown learning policy.')

      return learning_rate


def get_label_weight_mask(labels, ignore_label=255):
    """Gets the label weight mask.

      Args:
        labels: A Tensor of labels with the shape of [-1].
        ignore_label: Integer, label to ignore.
      Returns:
        A Tensor of label weights with the same shape of labels, each element is the
        weight for the label with the same index in labels,
        all valid labels have the same weight, and the element is 0.0
        if the label is to ignore.

    """

    not_ignore_mask = tf.not_equal(labels, ignore_label)
    not_ignore_mask = tf.cast(not_ignore_mask, tf.float32)

    return not_ignore_mask




