
nets
﻿
JSON.stringify(params.dense0)
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

JSON.stringify(params.dense1)
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

JSON.stringify(params.dense2)
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

JSON.stringify(params.dense3)
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

JSON.stringify(params.fc)
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
