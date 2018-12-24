import keras
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import sys

if len(sys.argv[1:]) > 0:
    modelName = sys.argv[1]
else:
    modelName = "pnasnet"

with tf.Graph().as_default():
  print("building module")
  detector = hub.Module(
      "https://tfhub.dev/google/imagenet/pnasnet_large/classification/2"
  )
  print("done building module")
  image_string_placeholder = tf.placeholder(tf.string)
  decoded_image = tf.image.decode_jpeg('gorge.jpg')
  # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
  # of size 1 and type tf.float32.
  decoded_image_float = tf.image.convert_image_dtype(
      image=decoded_image, dtype=tf.float32)
  module_input = tf.expand_dims(decoded_image_float, 0)
  result = detector(module_input, as_dict=True)
  init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

  session = tf.Session()
  print("running session")
  session.run(init_ops)
  print("done running session")

  # Use TF to save the graph model instead of Keras save model to load it in Golang
  builder = tf.saved_model.builder.SavedModelBuilder(modelName)
  # Tag the model, required for Go
  builder.add_meta_graph_and_variables(session, ["myTag"])
  builder.save()
  session.close()
