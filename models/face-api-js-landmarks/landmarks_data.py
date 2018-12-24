import json

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
