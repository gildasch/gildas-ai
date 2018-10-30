package main

import (
	"fmt"
	goimage "image"
	_ "image/jpeg"
	"io/ioutil"
	"os"

	"github.com/nfnt/resize"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
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

// Convert the image in filename to a Tensor suitable as input to the Inception model.
func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}
	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	// Some constants specific to the pre-trained model at:
	// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
	//
	// - The model was trained after with images scaled to 224x224 pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.
	const (
		H, W  = 299, 299
		Mean  = float32(117)
		Scale = float32(1)
	)
	// - input is a String-Tensor, where the string the JPEG-encoded image.
	// - The inception model takes a 4D tensor of shape
	//   [BatchSize, Height, Width, Colors=3], where each pixel is
	//   represented as a triplet of floats
	// - Apply normalization on each pixel and use ExpandDims to make
	//   this single image be a "batch" of size 1 for ResizeBilinear.
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ResizeBilinear(s,
		op.ExpandDims(s,
			op.Cast(s,
				op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
			op.Const(s.SubScope("make_batch"), int32(0))),
		op.Const(s.SubScope("size"), []int32{H, W}))
	graph, err = s.Finalize()
	return graph, input, output, err
}
