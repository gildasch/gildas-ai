const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var fs = require('fs');

var x = tf.tensor(JSON.parse(fs.readFileSync('normalized.json', 'utf8'))).reshape([1,112,112,3])
var filters = tf.tensor(JSON.parse(fs.readFileSync('filters.json', 'utf8'))).reshape([3,3,3,32])
var bias = tf.tensor(JSON.parse(fs.readFileSync('bias.json', 'utf8')))

var out = tf.add(
  tf.conv2d(x, filters, [2, 2], 'same'),
  bias)

out.print()
