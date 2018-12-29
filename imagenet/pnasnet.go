package imagenet

func NewPnasnet(modelRoot string) (*Model, func() error, error) {
	model := &Model{
		ID:              "pnasnet_tf_1.8.0",
		ModelName:       modelRoot + "models/tfhub_imagenet_pnasnet_large_classification/pnasnet_tf_1.8.0",
		TagName:         "myTag",
		InputLayer:      "module/hub_input/images",
		OutputLayer:     "module/final_layer/predictions",
		ImageMode:       ImageModeTensorflowPositive,
		Labels:          DefaultLabels,
		ImageHeight:     331,
		ImageWidth:      331,
		IndexCorrection: -1,
	}
	close, err := model.Load()
	if err != nil {
		return nil, nil, err
	}

	return model, close, nil
}
