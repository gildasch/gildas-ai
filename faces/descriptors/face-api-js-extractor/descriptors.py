import tensorflow as tf
import json
import numpy

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

normalized = loadAnNumpyArray("normalized.json", (1,150,150,3), numpy.float32)

params["conv32_down"] = {}
params["conv32_down"]["conv"] = {}
params["conv32_down"]["conv"]["filters"] = load("conv32_down.conv.filters.json", [7,7,3,32], tf.float32)
params["conv32_down"]["conv"]["bias"] = load("conv32_down.conv.bias.json", [32], tf.float32)
params["conv32_down"]["scale"] = {}
params["conv32_down"]["scale"]["biases"] = load("conv32_down.scale.biases.json", [32], tf.float32)
params["conv32_down"]["scale"]["weights"] = load("conv32_down.scale.weights.json", [32], tf.float32)

params["conv32_1"] = {}
params["conv32_1"]["conv1"] = {}
params["conv32_1"]["conv1"]["conv"] = {}
params["conv32_1"]["conv1"]["conv"]["filters"] = load("conv32_1.conv1.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_1"]["conv1"]["conv"]["bias"] = load("conv32_1.conv1.conv.bias.json", [32], tf.float32)
params["conv32_1"]["conv1"]["scale"] = {}
params["conv32_1"]["conv1"]["scale"]["biases"] = load("conv32_1.conv1.scale.biases.json", [32], tf.float32)
params["conv32_1"]["conv1"]["scale"]["weights"] = load("conv32_1.conv1.scale.weights.json", [32], tf.float32)
params["conv32_1"]["conv2"] = {}
params["conv32_1"]["conv2"]["conv"] = {}
params["conv32_1"]["conv2"]["conv"]["filters"] = load("conv32_1.conv2.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_1"]["conv2"]["conv"]["bias"] = load("conv32_1.conv2.conv.bias.json", [32], tf.float32)
params["conv32_1"]["conv2"]["scale"] = {}
params["conv32_1"]["conv2"]["scale"]["biases"] = load("conv32_1.conv2.scale.biases.json", [32], tf.float32)
params["conv32_1"]["conv2"]["scale"]["weights"] = load("conv32_1.conv2.scale.weights.json", [32], tf.float32)

params["conv32_2"] = {}
params["conv32_2"]["conv1"] = {}
params["conv32_2"]["conv1"]["conv"] = {}
params["conv32_2"]["conv1"]["conv"]["filters"] = load("conv32_2.conv1.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_2"]["conv1"]["conv"]["bias"] = load("conv32_2.conv1.conv.bias.json", [32], tf.float32)
params["conv32_2"]["conv1"]["scale"] = {}
params["conv32_2"]["conv1"]["scale"]["biases"] = load("conv32_2.conv1.scale.biases.json", [32], tf.float32)
params["conv32_2"]["conv1"]["scale"]["weights"] = load("conv32_2.conv1.scale.weights.json", [32], tf.float32)
params["conv32_2"]["conv2"] = {}
params["conv32_2"]["conv2"]["conv"] = {}
params["conv32_2"]["conv2"]["conv"]["filters"] = load("conv32_2.conv2.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_2"]["conv2"]["conv"]["bias"] = load("conv32_2.conv2.conv.bias.json", [32], tf.float32)
params["conv32_2"]["conv2"]["scale"] = {}
params["conv32_2"]["conv2"]["scale"]["biases"] = load("conv32_2.conv2.scale.biases.json", [32], tf.float32)
params["conv32_2"]["conv2"]["scale"]["weights"] = load("conv32_2.conv2.scale.weights.json", [32], tf.float32)

params["conv32_3"] = {}
params["conv32_3"]["conv1"] = {}
params["conv32_3"]["conv1"]["conv"] = {}
params["conv32_3"]["conv1"]["conv"]["filters"] = load("conv32_3.conv1.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_3"]["conv1"]["conv"]["bias"] = load("conv32_3.conv1.conv.bias.json", [32], tf.float32)
params["conv32_3"]["conv1"]["scale"] = {}
params["conv32_3"]["conv1"]["scale"]["biases"] = load("conv32_3.conv1.scale.biases.json", [32], tf.float32)
params["conv32_3"]["conv1"]["scale"]["weights"] = load("conv32_3.conv1.scale.weights.json", [32], tf.float32)
params["conv32_3"]["conv2"] = {}
params["conv32_3"]["conv2"]["conv"] = {}
params["conv32_3"]["conv2"]["conv"]["filters"] = load("conv32_3.conv2.conv.filters.json", [3,3,32,32], tf.float32)
params["conv32_3"]["conv2"]["conv"]["bias"] = load("conv32_3.conv2.conv.bias.json", [32], tf.float32)
params["conv32_3"]["conv2"]["scale"] = {}
params["conv32_3"]["conv2"]["scale"]["biases"] = load("conv32_3.conv2.scale.biases.json", [32], tf.float32)
params["conv32_3"]["conv2"]["scale"]["weights"] = load("conv32_3.conv2.scale.weights.json", [32], tf.float32)

params["conv64_down"] = {}
params["conv64_down"]["conv1"] = {}
params["conv64_down"]["conv1"]["conv"] = {}
params["conv64_down"]["conv1"]["conv"]["filters"] = load("conv64_down.conv1.conv.filters.json", [3,3,32,64], tf.float32)
params["conv64_down"]["conv1"]["conv"]["bias"] = load("conv64_down.conv1.conv.bias.json", [64], tf.float32)
params["conv64_down"]["conv1"]["scale"] = {}
params["conv64_down"]["conv1"]["scale"]["biases"] = load("conv64_down.conv1.scale.biases.json", [64], tf.float32)
params["conv64_down"]["conv1"]["scale"]["weights"] = load("conv64_down.conv1.scale.weights.json", [64], tf.float32)
params["conv64_down"]["conv2"] = {}
params["conv64_down"]["conv2"]["conv"] = {}
params["conv64_down"]["conv2"]["conv"]["filters"] = load("conv64_down.conv2.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_down"]["conv2"]["conv"]["bias"] = load("conv64_down.conv2.conv.bias.json", [64], tf.float32)
params["conv64_down"]["conv2"]["scale"] = {}
params["conv64_down"]["conv2"]["scale"]["biases"] = load("conv64_down.conv2.scale.biases.json", [64], tf.float32)
params["conv64_down"]["conv2"]["scale"]["weights"] = load("conv64_down.conv2.scale.weights.json", [64], tf.float32)

params["conv64_1"] = {}
params["conv64_1"]["conv1"] = {}
params["conv64_1"]["conv1"]["conv"] = {}
params["conv64_1"]["conv1"]["conv"]["filters"] = load("conv64_1.conv1.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_1"]["conv1"]["conv"]["bias"] = load("conv64_1.conv1.conv.bias.json", [64], tf.float32)
params["conv64_1"]["conv1"]["scale"] = {}
params["conv64_1"]["conv1"]["scale"]["biases"] = load("conv64_1.conv1.scale.biases.json", [64], tf.float32)
params["conv64_1"]["conv1"]["scale"]["weights"] = load("conv64_1.conv1.scale.weights.json", [64], tf.float32)
params["conv64_1"]["conv2"] = {}
params["conv64_1"]["conv2"]["conv"] = {}
params["conv64_1"]["conv2"]["conv"]["filters"] = load("conv64_1.conv2.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_1"]["conv2"]["conv"]["bias"] = load("conv64_1.conv2.conv.bias.json", [64], tf.float32)
params["conv64_1"]["conv2"]["scale"] = {}
params["conv64_1"]["conv2"]["scale"]["biases"] = load("conv64_1.conv2.scale.biases.json", [64], tf.float32)
params["conv64_1"]["conv2"]["scale"]["weights"] = load("conv64_1.conv2.scale.weights.json", [64], tf.float32)

params["conv64_2"] = {}
params["conv64_2"]["conv1"] = {}
params["conv64_2"]["conv1"]["conv"] = {}
params["conv64_2"]["conv1"]["conv"]["filters"] = load("conv64_2.conv1.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_2"]["conv1"]["conv"]["bias"] = load("conv64_2.conv1.conv.bias.json", [64], tf.float32)
params["conv64_2"]["conv1"]["scale"] = {}
params["conv64_2"]["conv1"]["scale"]["biases"] = load("conv64_2.conv1.scale.biases.json", [64], tf.float32)
params["conv64_2"]["conv1"]["scale"]["weights"] = load("conv64_2.conv1.scale.weights.json", [64], tf.float32)
params["conv64_2"]["conv2"] = {}
params["conv64_2"]["conv2"]["conv"] = {}
params["conv64_2"]["conv2"]["conv"]["filters"] = load("conv64_2.conv2.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_2"]["conv2"]["conv"]["bias"] = load("conv64_2.conv2.conv.bias.json", [64], tf.float32)
params["conv64_2"]["conv2"]["scale"] = {}
params["conv64_2"]["conv2"]["scale"]["biases"] = load("conv64_2.conv2.scale.biases.json", [64], tf.float32)
params["conv64_2"]["conv2"]["scale"]["weights"] = load("conv64_2.conv2.scale.weights.json", [64], tf.float32)

params["conv64_3"] = {}
params["conv64_3"]["conv1"] = {}
params["conv64_3"]["conv1"]["conv"] = {}
params["conv64_3"]["conv1"]["conv"]["filters"] = load("conv64_3.conv1.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_3"]["conv1"]["conv"]["bias"] = load("conv64_3.conv1.conv.bias.json", [64], tf.float32)
params["conv64_3"]["conv1"]["scale"] = {}
params["conv64_3"]["conv1"]["scale"]["biases"] = load("conv64_3.conv1.scale.biases.json", [64], tf.float32)
params["conv64_3"]["conv1"]["scale"]["weights"] = load("conv64_3.conv1.scale.weights.json", [64], tf.float32)
params["conv64_3"]["conv2"] = {}
params["conv64_3"]["conv2"]["conv"] = {}
params["conv64_3"]["conv2"]["conv"]["filters"] = load("conv64_3.conv2.conv.filters.json", [3,3,64,64], tf.float32)
params["conv64_3"]["conv2"]["conv"]["bias"] = load("conv64_3.conv2.conv.bias.json", [64], tf.float32)
params["conv64_3"]["conv2"]["scale"] = {}
params["conv64_3"]["conv2"]["scale"]["biases"] = load("conv64_3.conv2.scale.biases.json", [64], tf.float32)
params["conv64_3"]["conv2"]["scale"]["weights"] = load("conv64_3.conv2.scale.weights.json", [64], tf.float32)

params["conv128_down"] = {}
params["conv128_down"]["conv1"] = {}
params["conv128_down"]["conv1"]["conv"] = {}
params["conv128_down"]["conv1"]["conv"]["filters"] = load("conv128_down.conv1.conv.filters.json", [3,3,64,128], tf.float32)
params["conv128_down"]["conv1"]["conv"]["bias"] = load("conv128_down.conv1.conv.bias.json", [128], tf.float32)
params["conv128_down"]["conv1"]["scale"] = {}
params["conv128_down"]["conv1"]["scale"]["biases"] = load("conv128_down.conv1.scale.biases.json", [128], tf.float32)
params["conv128_down"]["conv1"]["scale"]["weights"] = load("conv128_down.conv1.scale.weights.json", [128], tf.float32)
params["conv128_down"]["conv2"] = {}
params["conv128_down"]["conv2"]["conv"] = {}
params["conv128_down"]["conv2"]["conv"]["filters"] = load("conv128_down.conv2.conv.filters.json", [3,3,128,128], tf.float32)
params["conv128_down"]["conv2"]["conv"]["bias"] = load("conv128_down.conv2.conv.bias.json", [128], tf.float32)
params["conv128_down"]["conv2"]["scale"] = {}
params["conv128_down"]["conv2"]["scale"]["biases"] = load("conv128_down.conv2.scale.biases.json", [128], tf.float32)
params["conv128_down"]["conv2"]["scale"]["weights"] = load("conv128_down.conv2.scale.weights.json", [128], tf.float32)

params["conv128_1"] = {}
params["conv128_1"]["conv1"] = {}
params["conv128_1"]["conv1"]["conv"] = {}
params["conv128_1"]["conv1"]["conv"]["filters"] = load("conv128_1.conv1.conv.filters.json", [3,3,128,128], tf.float32)
params["conv128_1"]["conv1"]["conv"]["bias"] = load("conv128_1.conv1.conv.bias.json", [128], tf.float32)
params["conv128_1"]["conv1"]["scale"] = {}
params["conv128_1"]["conv1"]["scale"]["biases"] = load("conv128_1.conv1.scale.biases.json", [128], tf.float32)
params["conv128_1"]["conv1"]["scale"]["weights"] = load("conv128_1.conv1.scale.weights.json", [128], tf.float32)
params["conv128_1"]["conv2"] = {}
params["conv128_1"]["conv2"]["conv"] = {}
params["conv128_1"]["conv2"]["conv"]["filters"] = load("conv128_1.conv2.conv.filters.json", [3,3,128,128], tf.float32)
params["conv128_1"]["conv2"]["conv"]["bias"] = load("conv128_1.conv2.conv.bias.json", [128], tf.float32)
params["conv128_1"]["conv2"]["scale"] = {}
params["conv128_1"]["conv2"]["scale"]["biases"] = load("conv128_1.conv2.scale.biases.json", [128], tf.float32)
params["conv128_1"]["conv2"]["scale"]["weights"] = load("conv128_1.conv2.scale.weights.json", [128], tf.float32)

params["conv128_2"] = {}
params["conv128_2"]["conv1"] = {}
params["conv128_2"]["conv1"]["conv"] = {}
params["conv128_2"]["conv1"]["conv"]["filters"] = load("conv128_2.conv1.conv.filters.json", [3,3,128,128], tf.float32)
params["conv128_2"]["conv1"]["conv"]["bias"] = load("conv128_2.conv1.conv.bias.json", [128], tf.float32)
params["conv128_2"]["conv1"]["scale"] = {}
params["conv128_2"]["conv1"]["scale"]["biases"] = load("conv128_2.conv1.scale.biases.json", [128], tf.float32)
params["conv128_2"]["conv1"]["scale"]["weights"] = load("conv128_2.conv1.scale.weights.json", [128], tf.float32)
params["conv128_2"]["conv2"] = {}
params["conv128_2"]["conv2"]["conv"] = {}
params["conv128_2"]["conv2"]["conv"]["filters"] = load("conv128_2.conv2.conv.filters.json", [3,3,128,128], tf.float32)
params["conv128_2"]["conv2"]["conv"]["bias"] = load("conv128_2.conv2.conv.bias.json", [128], tf.float32)
params["conv128_2"]["conv2"]["scale"] = {}
params["conv128_2"]["conv2"]["scale"]["biases"] = load("conv128_2.conv2.scale.biases.json", [128], tf.float32)
params["conv128_2"]["conv2"]["scale"]["weights"] = load("conv128_2.conv2.scale.weights.json", [128], tf.float32)

params["conv256_down"] = {}
params["conv256_down"]["conv1"] = {}
params["conv256_down"]["conv1"]["conv"] = {}
params["conv256_down"]["conv1"]["conv"]["filters"] = load("conv256_down.conv1.conv.filters.json", [3,3,128,256], tf.float32)
params["conv256_down"]["conv1"]["conv"]["bias"] = load("conv256_down.conv1.conv.bias.json", [256], tf.float32)
params["conv256_down"]["conv1"]["scale"] = {}
params["conv256_down"]["conv1"]["scale"]["biases"] = load("conv256_down.conv1.scale.biases.json", [256], tf.float32)
params["conv256_down"]["conv1"]["scale"]["weights"] = load("conv256_down.conv1.scale.weights.json", [256], tf.float32)
params["conv256_down"]["conv2"] = {}
params["conv256_down"]["conv2"]["conv"] = {}
params["conv256_down"]["conv2"]["conv"]["filters"] = load("conv256_down.conv2.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_down"]["conv2"]["conv"]["bias"] = load("conv256_down.conv2.conv.bias.json", [256], tf.float32)
params["conv256_down"]["conv2"]["scale"] = {}
params["conv256_down"]["conv2"]["scale"]["biases"] = load("conv256_down.conv2.scale.biases.json", [256], tf.float32)
params["conv256_down"]["conv2"]["scale"]["weights"] = load("conv256_down.conv2.scale.weights.json", [256], tf.float32)

params["conv256_1"] = {}
params["conv256_1"]["conv1"] = {}
params["conv256_1"]["conv1"]["conv"] = {}
params["conv256_1"]["conv1"]["conv"]["filters"] = load("conv256_1.conv1.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_1"]["conv1"]["conv"]["bias"] = load("conv256_1.conv1.conv.bias.json", [256], tf.float32)
params["conv256_1"]["conv1"]["scale"] = {}
params["conv256_1"]["conv1"]["scale"]["biases"] = load("conv256_1.conv1.scale.biases.json", [256], tf.float32)
params["conv256_1"]["conv1"]["scale"]["weights"] = load("conv256_1.conv1.scale.weights.json", [256], tf.float32)
params["conv256_1"]["conv2"] = {}
params["conv256_1"]["conv2"]["conv"] = {}
params["conv256_1"]["conv2"]["conv"]["filters"] = load("conv256_1.conv2.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_1"]["conv2"]["conv"]["bias"] = load("conv256_1.conv2.conv.bias.json", [256], tf.float32)
params["conv256_1"]["conv2"]["scale"] = {}
params["conv256_1"]["conv2"]["scale"]["biases"] = load("conv256_1.conv2.scale.biases.json", [256], tf.float32)
params["conv256_1"]["conv2"]["scale"]["weights"] = load("conv256_1.conv2.scale.weights.json", [256], tf.float32)

params["conv256_2"] = {}
params["conv256_2"]["conv1"] = {}
params["conv256_2"]["conv1"]["conv"] = {}
params["conv256_2"]["conv1"]["conv"]["filters"] = load("conv256_2.conv1.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_2"]["conv1"]["conv"]["bias"] = load("conv256_2.conv1.conv.bias.json", [256], tf.float32)
params["conv256_2"]["conv1"]["scale"] = {}
params["conv256_2"]["conv1"]["scale"]["biases"] = load("conv256_2.conv1.scale.biases.json", [256], tf.float32)
params["conv256_2"]["conv1"]["scale"]["weights"] = load("conv256_2.conv1.scale.weights.json", [256], tf.float32)
params["conv256_2"]["conv2"] = {}
params["conv256_2"]["conv2"]["conv"] = {}
params["conv256_2"]["conv2"]["conv"]["filters"] = load("conv256_2.conv2.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_2"]["conv2"]["conv"]["bias"] = load("conv256_2.conv2.conv.bias.json", [256], tf.float32)
params["conv256_2"]["conv2"]["scale"] = {}
params["conv256_2"]["conv2"]["scale"]["biases"] = load("conv256_2.conv2.scale.biases.json", [256], tf.float32)
params["conv256_2"]["conv2"]["scale"]["weights"] = load("conv256_2.conv2.scale.weights.json", [256], tf.float32)

params["conv256_down_out"] = {}
params["conv256_down_out"]["conv1"] = {}
params["conv256_down_out"]["conv1"]["conv"] = {}
params["conv256_down_out"]["conv1"]["conv"]["filters"] = load("conv256_down_out.conv1.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_down_out"]["conv1"]["conv"]["bias"] = load("conv256_down_out.conv1.conv.bias.json", [256], tf.float32)
params["conv256_down_out"]["conv1"]["scale"] = {}
params["conv256_down_out"]["conv1"]["scale"]["biases"] = load("conv256_down_out.conv1.scale.biases.json", [256], tf.float32)
params["conv256_down_out"]["conv1"]["scale"]["weights"] = load("conv256_down_out.conv1.scale.weights.json", [256], tf.float32)
params["conv256_down_out"]["conv2"] = {}
params["conv256_down_out"]["conv2"]["conv"] = {}
params["conv256_down_out"]["conv2"]["conv"]["filters"] = load("conv256_down_out.conv2.conv.filters.json", [3,3,256,256], tf.float32)
params["conv256_down_out"]["conv2"]["conv"]["bias"] = load("conv256_down_out.conv2.conv.bias.json", [256], tf.float32)
params["conv256_down_out"]["conv2"]["scale"] = {}
params["conv256_down_out"]["conv2"]["scale"]["biases"] = load("conv256_down_out.conv2.scale.biases.json", [256], tf.float32)
params["conv256_down_out"]["conv2"]["scale"]["weights"] = load("conv256_down_out.conv2.scale.weights.json", [256], tf.float32)

params["fc"] = load("fc.json", [256,128], tf.float32)

def scale(inp, params):
    return tf.math.add(
        tf.math.multiply(inp, params["weights"]),
        params["biases"])

def convLayer(inp, params, strides, withRelu, padding = 'SAME'):
    out = tf.nn.conv2d(inp, params["conv"]["filters"], strides, padding)
    out = tf.math.add(out, params["conv"]["bias"])
    out = scale(out, params["scale"])
    if withRelu:
        out = tf.nn.relu(out)
    return out

def conv(inp, params):
    return convLayer(inp, params, [1,1,1,1], True)

def convNoRelu(inp, params):
    return convLayer(inp, params, [1,1,1,1], False)

def convDown(inp, params):
    return convLayer(inp, params, [1,2,2,1], True, 'VALID')

def residual(inp, params):
    out = conv(inp, params["conv1"])
    out = convNoRelu(out, params["conv2"])
    out = tf.math.add(out, inp)
    out = tf.nn.relu(out)
    return out

def residualDown(inp, params):
    out = convDown(inp, params["conv1"])
    out = convNoRelu(out, params["conv2"])

    pooled = tf.nn.avg_pool(inp, [1,2,2,1], [1,2,2,1], 'VALID')
    zeros = tf.zeros(tf.shape(pooled), tf.float32)
    isPad = tf.shape(pooled)[3] != tf.shape(out)[3]
    isAdjustShape = tf.shape(pooled)[1] != tf.shape(out)[1] or tf.shape(pooled)[2] != tf.shape(out)[2]

    if isAdjustShape:
        padShapeX = tf.shape(out).eval()
        padShapeX[1] = 1
        zerosW = tf.zeros(padShapeX)
        out = tf.concat([out, zerosW], 1)

        padShapeY = tf.shape(out).eval()
        padShapeY[2] = 1
        zerosH = tf.zeros(padShapeY)
        out = tf.concat([out, zerosH], 2)

    if isPad:
        pooled = tf.concat([pooled, zeros], 3)

    out = tf.math.add(pooled, out)
    out = tf.relu(out)
    return out

sess = tf.Session()

with sess.as_default():
    out = convDown(normalized, params["conv32_down"])
    out = tf.nn.max_pool(out, [1,3,3,1], [1,2,2,1], 'VALID')

    out = residual(out, params["conv32_1"])
    out = residual(out, params["conv32_2"])
    out = residual(out, params["conv32_3"])

    out = residualDown(out, params["conv64_down"])
    out = residual(out, params["conv64_1"])
    out = residual(out, params["conv64_2"])
    out = residual(out, params["conv64_3"])

    out = residualDown(out, params["conv128_down"])
    out = residual(out, params["conv128_1"])
    out = residual(out, params["conv128_2"])

    out = residualDown(out, params["conv256_down"])
    out = residual(out, params["conv256_1"])
    out = residual(out, params["conv256_2"])
    out = residualDown(out, params["conv256_down_out"])

    globalAvg = tf.math.reduce_mean(out, [1,2])
    out = tf.linalg.matmul(globalAvg, params["fc"])

sess.Close()
