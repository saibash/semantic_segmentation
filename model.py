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
r"""Provides model definition and helper functions.

It is a deep convolutional neural network learning system for semantic image segmentation
with the following features:
(1) The architecture goes through a series of down-sampling layers
that reduce the dimensionality along the spatial dimension.
Pretrain down-sampling net ==> DenseNet121
(2) This is followed by a series of up-sampling layers,
it increases the dimensionality along the spatial dimensions,
Pretrain up-sampling net ==> pix2pix
(3) The output layer has the same spatial dimensions as the original input image

See the following papers for more details:
    Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017).
    Segnet: A deep convolutional encoder-decoder architecture for image segmentation.
    IEEE transactions on pattern analysis and machine intelligence, 39(12), 2481-2495.

    Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017).
    Image-to-image translation with conditional adversarial networks.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from utils import utils


def unet_model(input_shape=[256, 256, 3], n_classes=19, down_sampling_trainable=False):
    # load pretrain down-sampling net

    base = keras.applications.DenseNet121(input_shape=input_shape,
                                          include_top=False,
                                          weights='imagenet')

    # select the final ReLU activation layer for each feature map size
    # 4, 8, 16, 32, and 64
    skip_names = ['conv1/relu',  # size 64*64
                  'pool2_relu',  # size 32*32
                  'pool3_relu',  # size 16*16
                  'pool4_relu',  # size 8*8
                  'relu'  # size 4*4
                  ]
    skip_outputs = [base.get_layer(name).output for name in skip_names]
    down_sampling = keras.Model(inputs=base.input, outputs=skip_outputs)
    down_sampling.trainable = down_sampling_trainable

    # Four up-sampling net, sizes: 4->8, 8->16, 16->32, 32->64
    up_sampling = [pix2pix.upsample(512, 3),
                   pix2pix.upsample(256, 3),
                   pix2pix.upsample(128, 3),
                   pix2pix.upsample(64, 3)]

    # define the input layer
    inputs = tf.math.divide(keras.layers.Input(shape=input_shape), 255.)
    # down-sample
    down = down_sampling(inputs)
    out = down[-1]
    # prepare skip-connections
    skips = reversed(down[:-1])
    # up-sample with skip-connections, choose the last layer at first 4 --> 8
    for up, skip in zip(up_sampling, skips):
        out = up(out)
        out = keras.layers.Concatenate()([out, skip])
    # complete U-net model, define the final transpose conv layer
    out = keras.layers.Conv2DTranspose(n_classes, 3, strides=2, padding='same',
                                       activation=keras.activations.softmax)(out)

    return keras.Model(inputs=inputs, outputs=out)


def keras_custom_loss_function(y_actual, y_predicted, ignore_label=255):
    """return custom loss value.

          Returns:
          custom categorical cross entropy loss value,
          It ignores predictions for pixels with ignore_label value
          """
    label_weights = utils.get_label_weight_mask(y_actual, ignore_label=255)

    # Dimension of keep_mask is equal to the total number of pixels.
    keep_mask = tf.cast(tf.not_equal(y_actual, ignore_label), dtype=tf.float32)

    train_labels = tf.cast(y_actual, dtype=tf.float32) * keep_mask
    y_true = tf.one_hot(tf.cast(tf.squeeze(train_labels, axis=-1), dtype=tf.int32), depth=19)

    cce = tf.keras.losses.CategoricalCrossentropy()
    custom_loss_value = cce(y_true, y_predicted, sample_weight=label_weights)

    return custom_loss_value