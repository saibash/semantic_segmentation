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
import numpy as np
import matplotlib.pyplot as plt

import cv2


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


flags.DEFINE_list('crop_size', '256,256',
                  'Image crop size [height, width] during testing.')

flags.DEFINE_string('model_path', "my_h5model.h5",
                    'Trained model path.')



def show_inference(model, original_image, image_size):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #   image_np = np.array(Image.open(image_path))
    # Actual detection.

    image_reduced = utils.resize_image(original_image, image_size)
    image_reduced = tf.expand_dims(image_reduced, axis=0)

    predictions_reduced = model.predict(image_reduced)
    predictions_reduced = tf.squeeze(predictions_reduced, axis=0)
    predictions_reduced = tf.argmax(predictions_reduced, axis=-1)

    # Reverse the resizing and padding operations performed in preprocessing.
    # First, we slice the valid regions (i.e., remove padded region) and then
    # we resize the predictions back.
    original_image_shape = np.array(np.shape(original_image))

    predictions = tf.image.resize(tf.expand_dims(predictions_reduced, axis=-1),
                                  tf.convert_to_tensor(original_image_shape[:2], dtype=tf.int32),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    predictions = np.array(predictions, dtype=np.uint8)

    predictions_reduced = np.expand_dims(np.array(predictions_reduced, dtype=np.uint8), axis=-1)

    return predictions_reduced, predictions


def main(unused_argv):

    cap = cv2.VideoCapture(0)

    while 1:
        _, img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        model_fn = keras.models.load_model(FLAGS.model_path,
                                           custom_objects={"keras_custom_loss_function": model.keras_custom_loss_function})

        predictions_reduced, predictions = show_inference(model_fn, img, [int(sz) for sz in FLAGS.crop_size])

        final_pred = cv2.applyColorMap(predictions, cv2.COLORMAP_HSV)  # JET)
        cv2.imshow('Final_prediction', final_pred)

        final_pred_reduced = cv2.applyColorMap(predictions_reduced, cv2.COLORMAP_HSV)  # JET)
        cv2.imshow('Final_prediction_reduced', final_pred_reduced)

        cv2.imshow('Original img', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.compat.v1.app.run()
