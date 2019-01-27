package main

import (
	"fmt"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	model, err := tf.LoadSavedModel("myModel",
		[]string{"myTag"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	input, err := tf.NewTensor([1][][][3]float32{})
	if err != nil {
		log.Fatal(err)
	}

	output, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("input_1").Output(0): input,
		},
		[]tf.Output{
			model.Graph.Operation("predictions/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(output)
}
