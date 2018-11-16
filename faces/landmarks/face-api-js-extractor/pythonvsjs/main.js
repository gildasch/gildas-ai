const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var fs = require('fs');

function load(filename, shape) {
  return tf.tensor(JSON.parse(fs.readFileSync(filename, 'utf8'))).reshape(shape)
}

var normalized = load('normalized.json', [1,112,112,3])

var params = {}

params.dense0 = {}
params.dense0.conv0 = {}
params.dense0.conv0.filters = load("../dense0.conv0.filters.json", [3,3,3,32])
params.dense0.conv0.bias = load("../dense0.conv0.bias.json", [32])
params.dense0.conv1 = {}
params.dense0.conv1.depthwise_filter = load("../dense0.conv1.depthwise_filter.json", [3,3,32,1])
params.dense0.conv1.pointwise_filter = load("../dense0.conv1.pointwise_filter.json", [1,1,32,32])
params.dense0.conv1.bias = load("../dense0.conv1.bias.json", [32])
params.dense0.conv2 = {}
params.dense0.conv2.depthwise_filter = load("../dense0.conv2.depthwise_filter.json", [3,3,32,1])
params.dense0.conv2.pointwise_filter = load("../dense0.conv2.pointwise_filter.json", [1,1,32,32])
params.dense0.conv2.bias = load("../dense0.conv2.bias.json", [32])
params.dense0.conv3 = {}
params.dense0.conv3.depthwise_filter = load("../dense0.conv3.depthwise_filter.json", [3,3,32,1])
params.dense0.conv3.pointwise_filter = load("../dense0.conv3.pointwise_filter.json", [1,1,32,32])
params.dense0.conv3.bias = load("../dense0.conv3.bias.json", [32])

params.dense1 = {}
params.dense1.conv0 = {}
params.dense1.conv0.depthwise_filter = load("../dense1.conv0.depthwise_filter.json", [3,3,32,1])
params.dense1.conv0.pointwise_filter = load("../dense1.conv0.pointwise_filter.json", [1,1,32,64])
params.dense1.conv0.bias = load("../dense1.conv0.bias.json", [64])
params.dense1.conv1 = {}
params.dense1.conv1.depthwise_filter = load("../dense1.conv1.depthwise_filter.json", [3,3,64,1])
params.dense1.conv1.pointwise_filter = load("../dense1.conv1.pointwise_filter.json", [1,1,64,64])
params.dense1.conv1.bias = load("../dense1.conv1.bias.json", [64])
params.dense1.conv2 = {}
params.dense1.conv2.depthwise_filter = load("../dense1.conv2.depthwise_filter.json", [3,3,64,1])
params.dense1.conv2.pointwise_filter = load("../dense1.conv2.pointwise_filter.json", [1,1,64,64])
params.dense1.conv2.bias = load("../dense1.conv2.bias.json", [64])
params.dense1.conv3 = {}
params.dense1.conv3.depthwise_filter = load("../dense1.conv3.depthwise_filter.json", [3,3,64,1])
params.dense1.conv3.pointwise_filter = load("../dense1.conv3.pointwise_filter.json", [1,1,64,64])
params.dense1.conv3.bias = load("../dense1.conv3.bias.json", [64])

params.dense2 = {}
params.dense2.conv0 = {}
params.dense2.conv0.depthwise_filter = load("../dense2.conv0.depthwise_filter.json", [3,3,64,1])
params.dense2.conv0.pointwise_filter = load("../dense2.conv0.pointwise_filter.json", [1,1,64,128])
params.dense2.conv0.bias = load("../dense2.conv0.bias.json", [128])
params.dense2.conv1 = {}
params.dense2.conv1.depthwise_filter = load("../dense2.conv1.depthwise_filter.json", [3,3,128,1])
params.dense2.conv1.pointwise_filter = load("../dense2.conv1.pointwise_filter.json", [1,1,128,128])
params.dense2.conv1.bias = load("../dense2.conv1.bias.json", [128])
params.dense2.conv2 = {}
params.dense2.conv2.depthwise_filter = load("../dense2.conv2.depthwise_filter.json", [3,3,128,1])
params.dense2.conv2.pointwise_filter = load("../dense2.conv2.pointwise_filter.json", [1,1,128,128])
params.dense2.conv2.bias = load("../dense2.conv2.bias.json", [128])
params.dense2.conv3 = {}
params.dense2.conv3.depthwise_filter = load("../dense2.conv3.depthwise_filter.json", [3,3,128,1])
params.dense2.conv3.pointwise_filter = load("../dense2.conv3.pointwise_filter.json", [1,1,128,128])
params.dense2.conv3.bias = load("../dense2.conv3.bias.json", [128])

params.dense3 = {}
params.dense3.conv0 = {}
params.dense3.conv0.depthwise_filter = load("../dense3.conv0.depthwise_filter.json", [3,3,128,1])
params.dense3.conv0.pointwise_filter = load("../dense3.conv0.pointwise_filter.json", [1,1,128,256])
params.dense3.conv0.bias = load("../dense3.conv0.bias.json", [256])
params.dense3.conv1 = {}
params.dense3.conv1.depthwise_filter = load("../dense3.conv1.depthwise_filter.json", [3,3,256,1])
params.dense3.conv1.pointwise_filter = load("../dense3.conv1.pointwise_filter.json", [1,1,256,256])
params.dense3.conv1.bias = load("../dense3.conv1.bias.json", [256])
params.dense3.conv2 = {}
params.dense3.conv2.depthwise_filter = load("../dense3.conv2.depthwise_filter.json", [3,3,256,1])
params.dense3.conv2.pointwise_filter = load("../dense3.conv2.pointwise_filter.json", [1,1,256,256])
params.dense3.conv2.bias = load("../dense3.conv2.bias.json", [256])
params.dense3.conv3 = {}
params.dense3.conv3.depthwise_filter = load("../dense3.conv3.depthwise_filter.json", [3,3,256,1])
params.dense3.conv3.pointwise_filter = load("../dense3.conv3.pointwise_filter.json", [1,1,256,256])
params.dense3.conv3.bias = load("../dense3.conv3.bias.json", [256])

params.fc = {}
params.fc.weights = load("../weights.json", [256,136])
params.fc.bias = load("../bias.json", [136])


function denseBlock(x, denseBlockParams, isFirstLayer = false) {
  return tf.tidy(() => {
    const out1 = tf.relu(
      isFirstLayer
        ? tf.add(
          tf.conv2d(x, denseBlockParams.conv0.filters, [2, 2], 'same'),
          denseBlockParams.conv0.bias
        )
        : depthwiseSeparableConv(x, denseBlockParams.conv0, [2, 2])
    )
    const out2 = depthwiseSeparableConv(out1, denseBlockParams.conv1, [1, 1])

    const in3 = tf.relu(tf.add(out1, out2))
    const out3 = depthwiseSeparableConv(in3, denseBlockParams.conv2, [1, 1])

    const in4 = tf.relu(tf.add(out1, tf.add(out2, out3)))
    const out4 = depthwiseSeparableConv(in4, denseBlockParams.conv3, [1, 1])

    return tf.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4))))
  })
}

function depthwiseSeparableConv(x, params, stride) {
  return tf.tidy(() => {
    let out = tf.separableConv2d(x, params.depthwise_filter, params.pointwise_filter, stride, 'same')
    out = tf.add(out, params.bias)
    return out
  })
}

function fullyConnectedLayer(x, params) {
  return tf.tidy(() =>
    tf.add(
      tf.matMul(x, params.weights),
      params.bias
    )
  )
}

var out = tf.tidy(() => {
      // const batchTensor = input.toBatchTensor(112, true)
      // const meanRgb = [122.782, 117.001, 104.298]
      // const normalized = normalize(batchTensor, meanRgb).div(tf.scalar(255))

      let out = denseBlock(normalized, params.dense0, true)
      out = denseBlock(out, params.dense1)
      out = denseBlock(out, params.dense2)
      out = denseBlock(out, params.dense3)
      out = tf.avgPool(out, [7, 7], [2, 2], 'valid')

  return fullyConnectedLayer(out.as2D(out.shape[0], -1), params.fc)
})

out.print()
