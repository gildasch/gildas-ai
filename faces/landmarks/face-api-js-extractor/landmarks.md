return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true)
      const meanRgb = [122.782, 117.001, 104.298]
      const normalized = normalize(batchTensor, meanRgb).div(tf.scalar(255)) as tf.Tensor4D

      let out = denseBlock(normalized, params.dense0, true)
      out = denseBlock(out, params.dense1)
      out = denseBlock(out, params.dense2)
      out = denseBlock(out, params.dense3)
      out = tf.avgPool(out, [7, 7], [2, 2], 'valid')

  return fullyConnectedLayer(out.as2D(out.shape[0], -1), params.fc)
})

function denseBlock(
  x: tf.Tensor4D,
  denseBlockParams: DenseBlock4Params,
  isFirstLayer: boolean = false
): tf.Tensor4D {
  return tf.tidy(() => {
    const out1 = tf.relu(
      isFirstLayer
        ? tf.add(
          tf.conv2d(x, (denseBlockParams.conv0 as ConvParams).filters, [2, 2], 'same'),
          denseBlockParams.conv0.bias
        )
        : depthwiseSeparableConv(x, denseBlockParams.conv0 as SeparableConvParams, [2, 2])
    ) as tf.Tensor4D
    const out2 = depthwiseSeparableConv(out1, denseBlockParams.conv1, [1, 1])

    const in3 = tf.relu(tf.add(out1, out2)) as tf.Tensor4D
    const out3 = depthwiseSeparableConv(in3, denseBlockParams.conv2, [1, 1])

    const in4 = tf.relu(tf.add(out1, tf.add(out2, out3))) as tf.Tensor4D
    const out4 = depthwiseSeparableConv(in4, denseBlockParams.conv3, [1, 1])

    return tf.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4)))) as tf.Tensor4D
  })
}

export function depthwiseSeparableConv(
  x: tf.Tensor4D,
  params: SeparableConvParams,
  stride: [number, number]
): tf.Tensor4D {
  return tf.tidy(() => {
    let out = tf.separableConv2d(x, params.depthwise_filter, params.pointwise_filter, stride, 'same')
    out = tf.add(out, params.bias)
    return out
  })
}

export function fullyConnectedLayer(
  x: tf.Tensor2D,
  params: FCParams
): tf.Tensor2D {
  return tf.tidy(() =>
    tf.add(
      tf.matMul(x, params.weights),
      params.bias
    )
  )
}


### params.dense0

{
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
}

### params.dense1

{
  "conv0": {
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
      "id": 109,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        32,
        64
      ],
      "dtype": "float32",
      "size": 2048,
      "strides": [
        2048,
        2048,
        64
      ],
      "dataId": null,
      "id": 110,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        64
      ],
      "dtype": "float32",
      "size": 64,
      "strides": [],
      "dataId": null,
      "id": 111,
      "rankType": "1"
    }
  },
  "conv1": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        64,
        1
      ],
      "dtype": "float32",
      "size": 576,
      "strides": [
        192,
        64,
        1
      ],
      "dataId": null,
      "id": 112,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        64,
        64
      ],
      "dtype": "float32",
      "size": 4096,
      "strides": [
        4096,
        4096,
        64
      ],
      "dataId": null,
      "id": 113,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        64
      ],
      "dtype": "float32",
      "size": 64,
      "strides": [],
      "dataId": null,
      "id": 114,
      "rankType": "1"
    }
  },
  "conv2": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        64,
        1
      ],
      "dtype": "float32",
      "size": 576,
      "strides": [
        192,
        64,
        1
      ],
      "dataId": null,
      "id": 115,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        64,
        64
      ],
      "dtype": "float32",
      "size": 4096,
      "strides": [
        4096,
        4096,
        64
      ],
      "dataId": null,
      "id": 116,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        64
      ],
      "dtype": "float32",
      "size": 64,
      "strides": [],
      "dataId": null,
      "id": 117,
      "rankType": "1"
    }
  },
  "conv3": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        64,
        1
      ],
      "dtype": "float32",
      "size": 576,
      "strides": [
        192,
        64,
        1
      ],
      "dataId": null,
      "id": 118,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        64,
        64
      ],
      "dtype": "float32",
      "size": 4096,
      "strides": [
        4096,
        4096,
        64
      ],
      "dataId": null,
      "id": 119,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        64
      ],
      "dtype": "float32",
      "size": 64,
      "strides": [],
      "dataId": null,
      "id": 120,
      "rankType": "1"
    }
  }
}

### params.dense2

{
  "conv0": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        64,
        1
      ],
      "dtype": "float32",
      "size": 576,
      "strides": [
        192,
        64,
        1
      ],
      "dataId": null,
      "id": 121,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        64,
        128
      ],
      "dtype": "float32",
      "size": 8192,
      "strides": [
        8192,
        8192,
        128
      ],
      "dataId": null,
      "id": 122,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        128
      ],
      "dtype": "float32",
      "size": 128,
      "strides": [],
      "dataId": null,
      "id": 123,
      "rankType": "1"
    }
  },
  "conv1": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        128,
        1
      ],
      "dtype": "float32",
      "size": 1152,
      "strides": [
        384,
        128,
        1
      ],
      "dataId": null,
      "id": 124,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        128,
        128
      ],
      "dtype": "float32",
      "size": 16384,
      "strides": [
        16384,
        16384,
        128
      ],
      "dataId": null,
      "id": 125,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        128
      ],
      "dtype": "float32",
      "size": 128,
      "strides": [],
      "dataId": null,
      "id": 126,
      "rankType": "1"
    }
  },
  "conv2": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        128,
        1
      ],
      "dtype": "float32",
      "size": 1152,
      "strides": [
        384,
        128,
        1
      ],
      "dataId": null,
      "id": 127,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        128,
        128
      ],
      "dtype": "float32",
      "size": 16384,
      "strides": [
        16384,
        16384,
        128
      ],
      "dataId": null,
      "id": 128,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        128
      ],
      "dtype": "float32",
      "size": 128,
      "strides": [],
      "dataId": null,
      "id": 129,
      "rankType": "1"
    }
  },
  "conv3": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        128,
        1
      ],
      "dtype": "float32",
      "size": 1152,
      "strides": [
        384,
        128,
        1
      ],
      "dataId": null,
      "id": 130,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        128,
        128
      ],
      "dtype": "float32",
      "size": 16384,
      "strides": [
        16384,
        16384,
        128
      ],
      "dataId": null,
      "id": 131,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        128
      ],
      "dtype": "float32",
      "size": 128,
      "strides": [],
      "dataId": null,
      "id": 132,
      "rankType": "1"
    }
  }
}

### params.dense3

{
  "conv0": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        128,
        1
      ],
      "dtype": "float32",
      "size": 1152,
      "strides": [
        384,
        128,
        1
      ],
      "dataId": null,
      "id": 133,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        128,
        256
      ],
      "dtype": "float32",
      "size": 32768,
      "strides": [
        32768,
        32768,
        256
      ],
      "dataId": null,
      "id": 134,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        256
      ],
      "dtype": "float32",
      "size": 256,
      "strides": [],
      "dataId": null,
      "id": 135,
      "rankType": "1"
    }
  },
  "conv1": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        256,
        1
      ],
      "dtype": "float32",
      "size": 2304,
      "strides": [
        768,
        256,
        1
      ],
      "dataId": null,
      "id": 136,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        256,
        256
      ],
      "dtype": "float32",
      "size": 65536,
      "strides": [
        65536,
        65536,
        256
      ],
      "dataId": null,
      "id": 137,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        256
      ],
      "dtype": "float32",
      "size": 256,
      "strides": [],
      "dataId": null,
      "id": 138,
      "rankType": "1"
    }
  },
  "conv2": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        256,
        1
      ],
      "dtype": "float32",
      "size": 2304,
      "strides": [
        768,
        256,
        1
      ],
      "dataId": null,
      "id": 139,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        256,
        256
      ],
      "dtype": "float32",
      "size": 65536,
      "strides": [
        65536,
        65536,
        256
      ],
      "dataId": null,
      "id": 140,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        256
      ],
      "dtype": "float32",
      "size": 256,
      "strides": [],
      "dataId": null,
      "id": 141,
      "rankType": "1"
    }
  },
  "conv3": {
    "depthwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        3,
        3,
        256,
        1
      ],
      "dtype": "float32",
      "size": 2304,
      "strides": [
        768,
        256,
        1
      ],
      "dataId": null,
      "id": 142,
      "rankType": "4"
    },
    "pointwise_filter": {
      "isDisposedInternal": false,
      "shape": [
        1,
        1,
        256,
        256
      ],
      "dtype": "float32",
      "size": 65536,
      "strides": [
        65536,
        65536,
        256
      ],
      "dataId": null,
      "id": 143,
      "rankType": "4"
    },
    "bias": {
      "isDisposedInternal": false,
      "shape": [
        256
      ],
      "dtype": "float32",
      "size": 256,
      "strides": [],
      "dataId": null,
      "id": 144,
      "rankType": "1"
    }
  }
}

### params.fc

{
  "weights": {
    "isDisposedInternal": false,
    "shape": [
      256,
      136
    ],
    "dtype": "float32",
    "size": 34816,
    "strides": [
      136
    ],
    "dataId": null,
    "id": 145,
    "rankType": "2"
  },
  "bias": {
    "isDisposedInternal": false,
    "shape": [
      136
    ],
    "dtype": "float32",
    "size": 136,
    "strides": [],
    "dataId": null,
    "id": 146,
    "rankType": "1"
  }
}
