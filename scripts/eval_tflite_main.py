# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Eval checkpoint driver.

This is an example evaluation script for users to understand the EfficientNet
model checkpoints on CPU. To serve EfficientNet, please consider to export a
`SavedModel` from checkpoints and use tf-serving to serve.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import model_builder_factory
import preprocessing
import utils
import numpy as np
import argparse
from PIL import Image
import time
import pdb
import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file',
    help='.tflite model to be executed')
parser.add_argument(
    '-l',
    '--label_file',
    help='name of file containing labels')
parser.add_argument(
    '--num_threads', default=None, type=int, help='number of threads')
parser.add_argument('--iterations', default=50000, type=int)

args = parser.parse_args()

imagenet_eval_glob = '/data/ILSVRC2012_img_val/ILSVRC2012*.JPEG'
basedir = './tensors/'


def build_dataset(filenames, labels, is_training):
  """Build input dataset."""
  batch_drop_remainder = False
  filenames = tf.constant(filenames)
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

  def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    preprocess_fn = preprocessing.preprocess_image
    image_decoded = preprocess_fn(
        image_string, is_training, image_size=224)
    image = tf.cast(image_decoded, tf.float32)
    image -= tf.constant([127.0, 127.0, 127.0], shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant([128.0, 128.0, 128.0], shape=[1, 1, 3], dtype=image.dtype)
    return image, label
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(1,
                          drop_remainder=batch_drop_remainder)

  iterator = dataset.make_one_shot_iterator()
  return iterator

def main():
  if args.iterations == 1:
    interpreter = tf.lite.Interpreter(
        model_path=args.model_file, 
        num_threads=args.num_threads, 
        experimental_preserve_all_tensors=True)
  else:
    interpreter = tf.lite.Interpreter(
        model_path=args.model_file, 
        num_threads=args.num_threads, 
        experimental_preserve_all_tensors=False)
 
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print("== Input details ==")
  print("name:", input_details[0]['name'])
  print("shape:", input_details[0]['shape'])
  print("type:", input_details[0]['dtype'])
  print("\n== Output details ==")
  print("name:", output_details[0]['name'])
  print("shape:", output_details[0]['shape'])
  print("type:", output_details[0]['dtype'])

  floating_model = input_details[0]['dtype'] == np.float32
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  
  total_count = 0
  total_count5 = 0
  total_time = 0
  
  imagenet_val_labels = [int(i) for i in tf.gfile.GFile(args.label_file)]
  imagenet_filenames = sorted(tf.gfile.Glob(imagenet_eval_glob))
  image_files = imagenet_filenames[:args.iterations]
  labels = imagenet_val_labels[:args.iterations]
  iterator = build_dataset(image_files, labels, False)

  for i in range(args.iterations):
    images, labels = iterator.get_next()
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        images = images / input_scale + input_zero_point
        images = np.array(images).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]['index'], images)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)
    top_k = output_data.argsort()[::-1]
    total_count += labels in top_k[:1]
    total_count5 += labels in top_k[:5]
    total_time += stop_time - start_time
    num_images = i+1
    if (i + 1) % 10 == 0:
      print('iteration: {:5d}\tacc: {:4.2f}%\tacc-5: {:4.2f}%'.format(i+1, 100 * total_count / num_images, 100 * total_count5 / num_images))

  num_images = args.iterations
  print('acc: {:.5f}%\tacc-5: {:.5f}%\ttime: {}'.format( total_count / num_images, total_count5 / num_images, total_time / num_images))

  if args.iterations == 1:
    tensor_details = interpreter.get_tensor_details()
    nameset = dict()
    namefile = dict()
    for tdict in tensor_details:
      i = tdict['index']
      name = tdict['name']
      if len(name) == 0:
          continue
      scales = tdict['quantization_parameters']['scales']
      zero_points = tdict['quantization_parameters']['zero_points']
      # tensor is the np array with kernel weights or biases 
      tensor = interpreter.tensor(i)()
  
      #print(i, name, scales.shape, zero_points.shape, tensor.shape)
      filename = basedir + name.split(';')[-1].replace('/', '_')
  
      if filename not in nameset:
        nameset[filename] = 0
      nameset[filename] += 1
  
      filename = filename + '_{}'.format(nameset[filename])
      print(filename, zero_points[0:10], scales[0:10])
      np.save(filename + '_zeropoint', zero_points)
      np.save(filename + '_scales', scales)
      np.save(filename, tensor)
  
      namefile[filename] = name
    with open(basedir + 'layer_file_map.txt', 'w') as f:
      for k, v in namefile.items():
        f.write('{} {}\n'.format(v, k.split('/')[-1]))


if __name__ == '__main__':
  main()
