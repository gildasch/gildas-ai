package main

import (
	"fmt"
	goimage "image"
	_ "image/jpeg"
	"os"

	"github.com/nfnt/resize"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// replace myModel and myTag with the appropriate exported names in the chestrays-keras-binary-classification.ipynb
	model, err := tf.LoadSavedModel("myModel", []string{"myTag"}, nil)

	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	tensor, err := imageToTensor("gorge2.jpg")
	if err != nil {
		fmt.Println("cannot create tensor from image:", err)
		return
	}

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("input_1").Output(0): tensor, // Replace this with your input layer name
		},
		[]tf.Output{
			model.Graph.Operation("predictions/Softmax").Output(0), // Replace this with your output layer name
		},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running the session with input, err: %s\n", err.Error())
		return
	}

	for i := 0; i < 1; i++ {
		// fmt.Printf("Result value: %#+v \n", result[i].Value())
		res, ok := result[i].Value().([][]float32)
		if !ok {
			fmt.Println("missed conversion")
			return
		}

		best, bestV := 0, float32(0.0)
		for i, v := range res[0] {
			if bestV < v {
				best = i
				bestV = v
			}
		}

		fmt.Printf("Result value: %v (%f) \n", best, bestV)
	}
}

func imageToTensor(filename string) (*tf.Tensor, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	img, _, err := goimage.Decode(f)
	if err != nil {
		return nil, err
	}

	img = resize.Resize(299, 299, img, resize.NearestNeighbor)

	var image [1][299][299][3]float32
	for i := 0; i < 299; i++ {
		for j := 0; j < 299; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][i][j][0] = (float32(r>>8) - float32(127.5)) / float32(127.5)
			image[0][i][j][1] = (float32(g>>8) - float32(127.5)) / float32(127.5)
			image[0][i][j][2] = (float32(b>>8) - float32(127.5)) / float32(127.5)
			// fmt.Println(r, g, b, image[0][i][j][0], image[0][i][j][1], image[0][i][j][2])
			// return nil, nil
		}
	}

	return tf.NewTensor(image)
}
