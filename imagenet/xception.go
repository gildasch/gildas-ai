package imagenet

func NewXception(modelRoot string) (*Model, func() error, error) {
	model := &Model{
		ID:          "xception_tf_1.8.0",
		ModelName:   modelRoot + "models/harshsikka-Keras-Xception/xception_tf_1.8.0",
		TagName:     "myTag",
		InputLayer:  "input_1",
		OutputLayer: "predictions/Softmax",
		ImageMode:   ImageModeTensorflow,
		Labels:      DefaultLabels,
		ImageHeight: 299,
		ImageWidth:  299,
	}
	close, err := model.Load()
	if err != nil {
		return nil, nil, err
	}

	return model, close, nil
}
