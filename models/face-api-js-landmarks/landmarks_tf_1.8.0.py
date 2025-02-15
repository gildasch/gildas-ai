import tensorflow as tf
import json
import numpy
import os
import sys

params = {}

def load(filename, shape, dtype):
    with open(filename, 'r') as f:
        s = f.read()
        d = eval(s)
    t = tf.convert_to_tensor(d, dtype=dtype)
    t = tf.reshape(t, shape)
    return t

def loadAnNumpyArray(filename, shape, dtype):
    with open(filename, 'r') as f:
        s = f.read()
        d = eval(s)
    t = numpy.asarray(d, dtype=dtype)
    t = numpy.reshape(t, shape)
    return t

normalized = loadAnNumpyArray("normalized.json", (1,112,112,3), numpy.float32)

params["dense0"] = {}
params["dense0"]["conv0"] = {}
params["dense0"]["conv0"]["filters"] = load("dense0.conv0.filters.json", [3,3,3,32], tf.float32)
params["dense0"]["conv0"]["bias"] = load("dense0.conv0.bias.json", [32], tf.float32)
params["dense0"]["conv1"] = {}
params["dense0"]["conv1"]["depthwise_filter"] = load("dense0.conv1.depthwise_filter.json", [3,3,32,1], tf.float32)
params["dense0"]["conv1"]["pointwise_filter"] = load("dense0.conv1.pointwise_filter.json", [1,1,32,32], tf.float32)
params["dense0"]["conv1"]["bias"] = load("dense0.conv1.bias.json", [32], tf.float32)
params["dense0"]["conv2"] = {}
params["dense0"]["conv2"]["depthwise_filter"] = load("dense0.conv2.depthwise_filter.json", [3,3,32,1], tf.float32)
params["dense0"]["conv2"]["pointwise_filter"] = load("dense0.conv2.pointwise_filter.json", [1,1,32,32], tf.float32)
params["dense0"]["conv2"]["bias"] = load("dense0.conv2.bias.json", [32], tf.float32)
params["dense0"]["conv3"] = {}
params["dense0"]["conv3"]["depthwise_filter"] = load("dense0.conv3.depthwise_filter.json", [3,3,32,1], tf.float32)
params["dense0"]["conv3"]["pointwise_filter"] = load("dense0.conv3.pointwise_filter.json", [1,1,32,32], tf.float32)
params["dense0"]["conv3"]["bias"] = load("dense0.conv3.bias.json", [32], tf.float32)

params["dense1"] = {}
params["dense1"]["conv0"] = {}
params["dense1"]["conv0"]["depthwise_filter"] = load("dense1.conv0.depthwise_filter.json", [3,3,32,1], tf.float32)
params["dense1"]["conv0"]["pointwise_filter"] = load("dense1.conv0.pointwise_filter.json", [1,1,32,64], tf.float32)
params["dense1"]["conv0"]["bias"] = load("dense1.conv0.bias.json", [64], tf.float32)
params["dense1"]["conv1"] = {}
params["dense1"]["conv1"]["depthwise_filter"] = load("dense1.conv1.depthwise_filter.json", [3,3,64,1], tf.float32)
params["dense1"]["conv1"]["pointwise_filter"] = load("dense1.conv1.pointwise_filter.json", [1,1,64,64], tf.float32)
params["dense1"]["conv1"]["bias"] = load("dense1.conv1.bias.json", [64], tf.float32)
params["dense1"]["conv2"] = {}
params["dense1"]["conv2"]["depthwise_filter"] = load("dense1.conv2.depthwise_filter.json", [3,3,64,1], tf.float32)
params["dense1"]["conv2"]["pointwise_filter"] = load("dense1.conv2.pointwise_filter.json", [1,1,64,64], tf.float32)
params["dense1"]["conv2"]["bias"] = load("dense1.conv2.bias.json", [64], tf.float32)
params["dense1"]["conv3"] = {}
params["dense1"]["conv3"]["depthwise_filter"] = load("dense1.conv3.depthwise_filter.json", [3,3,64,1], tf.float32)
params["dense1"]["conv3"]["pointwise_filter"] = load("dense1.conv3.pointwise_filter.json", [1,1,64,64], tf.float32)
params["dense1"]["conv3"]["bias"] = load("dense1.conv3.bias.json", [64], tf.float32)

params["dense2"] = {}
params["dense2"]["conv0"] = {}
params["dense2"]["conv0"]["depthwise_filter"] = load("dense2.conv0.depthwise_filter.json", [3,3,64,1], tf.float32)
params["dense2"]["conv0"]["pointwise_filter"] = load("dense2.conv0.pointwise_filter.json", [1,1,64,128], tf.float32)
params["dense2"]["conv0"]["bias"] = load("dense2.conv0.bias.json", [128], tf.float32)
params["dense2"]["conv1"] = {}
params["dense2"]["conv1"]["depthwise_filter"] = load("dense2.conv1.depthwise_filter.json", [3,3,128,1], tf.float32)
params["dense2"]["conv1"]["pointwise_filter"] = load("dense2.conv1.pointwise_filter.json", [1,1,128,128], tf.float32)
params["dense2"]["conv1"]["bias"] = load("dense2.conv1.bias.json", [128], tf.float32)
params["dense2"]["conv2"] = {}
params["dense2"]["conv2"]["depthwise_filter"] = load("dense2.conv2.depthwise_filter.json", [3,3,128,1], tf.float32)
params["dense2"]["conv2"]["pointwise_filter"] = load("dense2.conv2.pointwise_filter.json", [1,1,128,128], tf.float32)
params["dense2"]["conv2"]["bias"] = load("dense2.conv2.bias.json", [128], tf.float32)
params["dense2"]["conv3"] = {}
params["dense2"]["conv3"]["depthwise_filter"] = load("dense2.conv3.depthwise_filter.json", [3,3,128,1], tf.float32)
params["dense2"]["conv3"]["pointwise_filter"] = load("dense2.conv3.pointwise_filter.json", [1,1,128,128], tf.float32)
params["dense2"]["conv3"]["bias"] = load("dense2.conv3.bias.json", [128], tf.float32)

params["dense3"] = {}
params["dense3"]["conv0"] = {}
params["dense3"]["conv0"]["depthwise_filter"] = load("dense3.conv0.depthwise_filter.json", [3,3,128,1], tf.float32)
params["dense3"]["conv0"]["pointwise_filter"] = load("dense3.conv0.pointwise_filter.json", [1,1,128,256], tf.float32)
params["dense3"]["conv0"]["bias"] = load("dense3.conv0.bias.json", [256], tf.float32)
params["dense3"]["conv1"] = {}
params["dense3"]["conv1"]["depthwise_filter"] = load("dense3.conv1.depthwise_filter.json", [3,3,256,1], tf.float32)
params["dense3"]["conv1"]["pointwise_filter"] = load("dense3.conv1.pointwise_filter.json", [1,1,256,256], tf.float32)
params["dense3"]["conv1"]["bias"] = load("dense3.conv1.bias.json", [256], tf.float32)
params["dense3"]["conv2"] = {}
params["dense3"]["conv2"]["depthwise_filter"] = load("dense3.conv2.depthwise_filter.json", [3,3,256,1], tf.float32)
params["dense3"]["conv2"]["pointwise_filter"] = load("dense3.conv2.pointwise_filter.json", [1,1,256,256], tf.float32)
params["dense3"]["conv2"]["bias"] = load("dense3.conv2.bias.json", [256], tf.float32)
params["dense3"]["conv3"] = {}
params["dense3"]["conv3"]["depthwise_filter"] = load("dense3.conv3.depthwise_filter.json", [3,3,256,1], tf.float32)
params["dense3"]["conv3"]["pointwise_filter"] = load("dense3.conv3.pointwise_filter.json", [1,1,256,256], tf.float32)
params["dense3"]["conv3"]["bias"] = load("dense3.conv3.bias.json", [256], tf.float32)

params["fc"] = {}
params["fc"]["weights"] = load("weights.json", [256,136], tf.float32)
params["fc"]["bias"] = load("bias.json", [136], tf.float32)

def denseLayer(inp, dense, isFirstLayer=False):
  if isFirstLayer:
    out1 = tf.add(
        tf.nn.conv2d(inp, dense["conv0"]["filters"], [1,2,2,1], 'SAME'),
        dense["conv0"]["bias"])
  else:
    out1 = tf.add(
      tf.nn.separable_conv2d(
        inp, dense["conv0"]["depthwise_filter"], dense["conv0"]["pointwise_filter"],
        [1,2,2,1], 'SAME'),
      dense["conv0"]["bias"])

  out1 = tf.nn.relu(out1)

  out2 = tf.add(
    tf.nn.separable_conv2d(
      out1, dense["conv1"]["depthwise_filter"], dense["conv1"]["pointwise_filter"],
      [1,1,1,1], 'SAME'),
    dense["conv1"]["bias"])

  out3 = tf.add(
    tf.nn.separable_conv2d(
      tf.nn.relu(tf.add(out1, out2)),
      dense["conv2"]["depthwise_filter"], dense["conv2"]["pointwise_filter"],
      [1,1,1,1], 'SAME'),
    dense["conv2"]["bias"])

  out4 = tf.add(
    tf.nn.separable_conv2d(
      tf.nn.relu(tf.add(out1, tf.add(out2, out3))),
      dense["conv3"]["depthwise_filter"], dense["conv3"]["pointwise_filter"],
      [1,1,1,1], 'SAME'),
    dense["conv3"]["bias"])

  return tf.nn.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4))))

inp = tf.placeholder(tf.float32, [1,112,112,3], name='input')

out = denseLayer(inp, params["dense0"], True)
out = denseLayer(out, params["dense1"])
out = denseLayer(out, params["dense2"])
out = denseLayer(out, params["dense3"])

out = tf.nn.avg_pool(out, [1,7,7,1], [1,2,2,1], 'VALID')

out = tf.add(
    tf.matmul(
        tf.reshape(out, [tf.shape(out)[0], -1]),
        params["fc"]["weights"]),
    params["fc"]["bias"], 'output')

if len(sys.argv[1:]) > 0:
    modelName = sys.argv[1]
else:
    modelName = "face-api-landmarksnet"

with tf.Session() as sess:
  print(sess.run(out, feed_dict={inp: normalized}))
  # Use TF to save the graph model instead of Keras save model to load it in Golang
  builder = tf.saved_model.builder.SavedModelBuilder(modelName)
  # Tag the model, required for Go
  builder.add_meta_graph_and_variables(sess, ["myTag"])
  builder.save()
  sess.close()
