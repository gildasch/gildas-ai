package imagenet

func NewResnet(modelRoot string) (*Model, func() error, error) {
	model := &Model{
		ID:          "resnet_tf_1.8.0",
		ModelName:   modelRoot + "models/tonyshih-Keras-ResNet50/resnet_tf_1.8.0",
		TagName:     "myTag",
		InputLayer:  "input_1",
		OutputLayer: "fc1000/Softmax",
		ImageMode:   ImageModeCaffe,
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
