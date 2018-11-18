public forwardInput(input: NetInput): tf.Tensor2D {

  const { params } = this

  if (!params) {
    throw new Error('FaceRecognitionNet - load model before inference')
  }

  return tf.tidy(() => {
    const batchTensor = input.toBatchTensor(150, true).toFloat()

    const meanRgb = [122.782, 117.001, 104.298]
    const normalized = normalize(batchTensor, meanRgb).div(tf.scalar(256)) as tf.Tensor4D

    let out = convDown(normalized, params.conv32_down)
    out = tf.maxPool(out, 3, 2, 'valid')

    out = residual(out, params.conv32_1)
    out = residual(out, params.conv32_2)
    out = residual(out, params.conv32_3)

    out = residualDown(out, params.conv64_down)
    out = residual(out, params.conv64_1)
    out = residual(out, params.conv64_2)
    out = residual(out, params.conv64_3)

    out = residualDown(out, params.conv128_down)
    out = residual(out, params.conv128_1)
    out = residual(out, params.conv128_2)

    out = residualDown(out, params.conv256_down)
    out = residual(out, params.conv256_1)
    out = residual(out, params.conv256_2)
    out = residualDown(out, params.conv256_down_out)

    const globalAvg = out.mean([1, 2]) as tf.Tensor2D
    const fullyConnected = tf.matMul(globalAvg, params.fc)

    return fullyConnected
    })
}

function convLayer(
  x: tf.Tensor4D,
  params: ConvLayerParams,
  strides: [number, number],
  withRelu: boolean,
  padding: 'valid' | 'same' = 'same'
): tf.Tensor4D {
  const { filters, bias } = params.conv

  let out = tf.conv2d(x, filters, strides, padding)
  out = tf.add(out, bias)
  out = scale(out, params.scale)
  return withRelu ? tf.relu(out) : out
}

export function conv(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [1, 1], true)
}

export function convNoRelu(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [1, 1], false)
}

export function convDown(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [2, 2], true, 'valid')
}

export function scale(x: tf.Tensor4D, params: ScaleLayerParams): tf.Tensor4D {
  return tf.add(tf.mul(x, params.weights), params.biases)
}

export function residual(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  let out = conv(x, params.conv1)
  out = convNoRelu(out, params.conv2)
  out = tf.add(out, x)
  out = tf.relu(out)
  return out
}

export function residualDown(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  let out = convDown(x, params.conv1)
  out = convNoRelu(out, params.conv2)

  let pooled = tf.avgPool(x, 2, 2, 'valid') as tf.Tensor4D
  const zeros = tf.zeros<tf.Rank.R4>(pooled.shape)
  const isPad = pooled.shape[3] !== out.shape[3]
  const isAdjustShape = pooled.shape[1] !== out.shape[1] || pooled.shape[2] !== out.shape[2]

  if (isAdjustShape) {
    const padShapeX = [...out.shape] as [number, number, number, number]
    padShapeX[1] = 1
    const zerosW = tf.zeros<tf.Rank.R4>(padShapeX)
    out = tf.concat([out, zerosW], 1)

    const padShapeY = [...out.shape] as [number, number, number, number]
    padShapeY[2] = 1
    const zerosH = tf.zeros<tf.Rank.R4>(padShapeY)
    out = tf.concat([out, zerosH], 2)
  }

  pooled = isPad ? tf.concat([pooled, zeros], 3) : pooled
  out = tf.add(pooled, out) as tf.Tensor4D

  out = tf.relu(out)
  return out
}
