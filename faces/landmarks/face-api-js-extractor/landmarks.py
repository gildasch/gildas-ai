import tensorflow as tf
import json

inp = [0]
dense0 = json.loads("""{
  "conv0": {
    "filters": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        3,
        32
      ],
      "dtype": "float32",
      "size": 864,
      "strides": [
        288,
        96,
        32
      ],
      "dataId": null,
      "id": 98,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        32
      ],
      "dtype": "float32",
      "size": 32,
      "strides": [],
      "dataId": null,
      "id": 99,
      "rankType": "1"
    }
  },
  "conv1": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        32,
        1
      ],
      "dtype": "float32",
      "size": 288,
      "strides": [
        96,
        32,
        1
      ],
      "dataId": null,
      "id": 100,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        32,
        32
      ],
      "dtype": "float32",
      "size": 1024,
      "strides": [
        1024,
        1024,
        32
      ],
      "dataId": null,
      "id": 101,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        32
      ],
      "dtype": "float32",
      "size": 32,
      "strides": [],
      "dataId": null,
      "id": 102,
      "rankType": "1"
    }
  },
  "conv2": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        32,
        1
      ],
      "dtype": "float32",
      "size": 288,
      "strides": [
        96,
        32,
        1
      ],
      "dataId": null,
      "id": 103,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        32,
        32
      ],
      "dtype": "float32",
      "size": 1024,
      "strides": [
        1024,
        1024,
        32
      ],
      "dataId": null,
      "id": 104,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        32
      ],
      "dtype": "float32",
      "size": 32,
      "strides": [],
      "dataId": null,
      "id": 105,
      "rankType": "1"
    }
  },
  "conv3": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        32,
        1
      ],
      "dtype": "float32",
      "size": 288,
      "strides": [
        96,
        32,
        1
      ],
      "dataId": null,
      "id": 106,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        32,
        32
      ],
      "dtype": "float32",
      "size": 1024,
      "strides": [
        1024,
        1024,
        32
      ],
      "dataId": null,
      "id": 107,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        32
      ],
      "dtype": "float32",
      "size": 32,
      "strides": [],
      "dataId": null,
      "id": 108,
      "rankType": "1"
    }
  }
}""")

out1 = tf.math.add(
    tf.nn.conv2d(inp, dense0["conv0"]["filters"], [2,2], 'SAME'),
    dense0["conv0"]["bias"])

out2 = tf.math.add(
    tf.nn.separableConv2d(
        out1, dense0["conv1"]["depthwise_filter"], dense0["conv1"]["pointwise_filter"],
        [1,1], 'SAME'),
    dense0["conv1"]["bias"])

out3 = tf.math.add(
    tf.nn.separableConv2d(
        tf.nn.relu(tf.math.add(out1, out2)),
        dense0["conv2"]["depthwise_filter"], dense0["conv2"]["pointwise_filter"],
        [1,1], 'SAME'),
    dense0["conv2"]["bias"])

out4 = tf.math.add(
    tf.nn.separableConv2d(
        tf.nn.relu(tf.math.add(out1, tf.math.add(out2, out3))),
        dense0["conv3"]["depthwise_filter"], dense0["conv3"]["pointwise_filter"],
        [1,1], 'SAME'),
    dense0["conv3"]["bias"])

outFinal = tf.nn.relu(tf.math.add(out1, tf.math.add(out2, tf.math.add(out3, out4))))



























# x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
# w = tf.Variable(tf.random_uniform([2, 2]))
# y = tf.matmul(x, w)
# output = tf.nn.softmax(y)
# init_op = w.initializer

# with tf.Session() as sess:
#   # Run the initializer on `w`.
#   sess.run(init_op)

#   # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
#   # the result of the computation.
#   print(sess.run(output))

#   # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
#   # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
#   # op. Both `y_val` and `output_val` will be NumPy arrays.
#   y_val, output_val = sess.run([y, output])

#   print(x)
#   print(w)
#   print(y_val)
#   print(output_val)
