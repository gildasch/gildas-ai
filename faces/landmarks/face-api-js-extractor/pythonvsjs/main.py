import tensorflow as tf
import json

def load(filename, shape, dtype):
    with open(filename, 'r') as f:
        s = f.read()
        d = eval(s)
    t = tf.convert_to_tensor(list(d), dtype=dtype)
    t = tf.reshape(t, shape)
    return t

x = load("normalized.json", [1,112,112,3], tf.float32)
filters = load("filters.json", [3,3,3,32], tf.float32)
bias = load("bias.json", [32], tf.float32)

out = tf.math.add(
  tf.nn.conv2d(x, filters, [1, 2, 2, 1], 'SAME'),
  bias)

with tf.Session() as sess:
  print(json.dumps(sess.run(out).tolist()))
