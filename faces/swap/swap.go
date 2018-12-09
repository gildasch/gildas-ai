package swap

import (
	"fmt"
	"image"
	"image/draw"

	"github.com/fogleman/gg"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/gildasch/gildas-ai/imageutils/distort"
	"github.com/nfnt/resize"
	"github.com/pkg/errors"
)

type Extractor interface {
	ExtractLandmarks(img image.Image) ([][]image.Point, []image.Image, error)
}

type LandmarkDetector interface {
	Detect(img image.Image) (*landmarks.Landmarks, error)
}

func FaceSwap(extractor Extractor, detector LandmarkDetector, dest, src image.Image) (image.Image, error) {
	srcLandmarks, srcCrops, err := extractor.ExtractLandmarks(src)
	if err != nil {
		return nil, errors.Wrap(err, "error extracting landmarks from src")
	}

	if len(srcLandmarks) == 0 {
		return nil, errors.New("no face detected in src image")
	}

	destLandmarks, destCrops, err := extractor.ExtractLandmarks(dest)
	if err != nil {
		return nil, errors.Wrap(err, "error extracting landmarks from dest")
	}

	for i := range destLandmarks {
		fmt.Println("bounds before", destCrops[i].Bounds())
		destCrops[i], err = swap(detector, srcCrops[0], destCrops[i])
		if err != nil {
			return nil, errors.Wrapf(err, "error swapping face %d", i)
		}
		fmt.Println("bounds after", destCrops[i].Bounds())

		gg.SavePNG(fmt.Sprintf("out-swap-%d.png", i), destCrops[i])
	}

	out := image.NewRGBA(dest.Bounds())
	draw.Draw(out, out.Bounds(), dest, dest.Bounds().Min, draw.Src)
	for _, swapped := range destCrops {
		fmt.Println("swapped bounds", swapped.Bounds())
		draw.Draw(out, out.Bounds(), swapped, image.ZP, draw.Over)
	}

	return out, nil
}

var counter = 1

func swap(detector LandmarkDetector, src, dest image.Image) (image.Image, error) {
	gg.SavePNG(fmt.Sprintf("out-swap-crop-src-%d.png", counter), src)
	gg.SavePNG(fmt.Sprintf("out-swap-crop-dest-%d.png", counter), dest)

	src = resize.Resize(uint(dest.Bounds().Dx()), uint(dest.Bounds().Dy()), src, resize.NearestNeighbor)

	srcLM, err := detector.Detect(src)
	if err != nil {
		return nil, err
	}
	srcLandmarks := srcLM.PointsOnImage(src)
	destLM, err := detector.Detect(dest)
	if err != nil {
		return nil, err
	}
	destLandmarks := destLM.PointsOnImage(dest)

	destLandmarksAlignedToZero := make([]image.Point, len(destLandmarks))
	for i, dl := range destLandmarks {
		destLandmarksAlignedToZero[i].X = dl.X - dest.Bounds().Min.X
		destLandmarksAlignedToZero[i].Y = dl.Y - dest.Bounds().Min.Y
	}
	distorted, err := distort.Distort(src, srcLandmarks, destLandmarksAlignedToZero)
	if err != nil {
		return nil, err
	}

	out := image.NewRGBA(dest.Bounds())
	draw.Draw(out, out.Bounds(), dest, dest.Bounds().Min, draw.Src)

	maskIn := image.NewRGBA(out.Bounds())
	mask := maskFromPolygon(maskIn, simplify(destLandmarks))
	maskAligned := image.NewRGBA(out.Bounds())
	draw.Draw(maskAligned, out.Bounds(), mask, image.ZP, draw.Src)

	gg.SavePNG(fmt.Sprintf("out-swap-mask-%d.png", counter), distorted)

	fmt.Println("out bounds", out.Bounds())
	fmt.Println("dest bounds", dest.Bounds())
	fmt.Println("mask bounds", maskAligned.Bounds())
	fmt.Println("distorted bounds", distorted.Bounds())
	draw.DrawMask(out, out.Bounds(), distorted, image.ZP, maskAligned, out.Bounds().Min, draw.Over)

	gg.SavePNG(fmt.Sprintf("out-swap-truc-%d.png", counter), out)
	counter++

	return out, nil
}

func maskFromPolygon(in image.Image, landmarks []image.Point) image.Image {
	fmt.Println("maskFromPolygon in.Bounds", in.Bounds())
	fmt.Println("maskFromPolygon landmarks", landmarks)
	dc := gg.NewContext(in.Bounds().Dx(), in.Bounds().Dy())

	for _, p := range landmarks {
		dc.LineTo(float64(p.X-in.Bounds().Min.X), float64(p.Y-in.Bounds().Min.Y))
	}

	dc.Fill()

	return dc.Image()
}

// 0 to 16 is the chin (8 is the tip of the chin)
// 26 to 22 is left eye brow
// 21 to 17 is right eye brow
func simplify(landmarks []image.Point) []image.Point {
	chin := landmarks[:17]

	for i := 26; i >= 17; i-- {
		chin = append(chin, landmarks[i])
	}

	return chin
}

const factor = 0.1

func moveUp(base, target image.Point) image.Point {
	target.X += int(factor * float32(target.X-base.X))
	target.Y += int(factor * float32(target.Y-base.Y))
	return target
}
