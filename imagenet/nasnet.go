package imagenet

func NewNasnet(modelRoot string) (*Model, func() error, error) {
	model := &Model{
		ID:          "nasnet-mobile_tf_" + tfVersion(),
		ModelName:   modelRoot + "models/jbrandowski_NASNet_Mobile/nasnet-mobile_tf_" + tfVersion(),
		TagName:     "myTag",
		InputLayer:  "input_1",
		OutputLayer: "predictions/Softmax",
		ImageMode:   ImageModeTensorflow,
		Labels:      DefaultLabels,
		ImageHeight: 224,
		ImageWidth:  224,
	}
	close, err := model.Load()
	if err != nil {
		return nil, nil, err
	}

	return model, close, nil
}
