package swap

import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"sort"

	"github.com/disintegration/imaging"
	"github.com/fogleman/gg"
	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils/distort"
	colorful "github.com/lucasb-eyer/go-colorful"
	"github.com/nfnt/resize"
	"github.com/pkg/errors"
)

type Extractor interface {
	ExtractLandmarks(img image.Image) ([][]image.Point, []image.Image, error)
}

type LandmarkDetector interface {
	Detect(img image.Image) (*gildasai.Landmarks, error)
}

func FaceSwap(extractor Extractor, detector LandmarkDetector, dest, src image.Image, blur float64) (image.Image, error) {
	srcLandmarks, srcCrops, err := extractor.ExtractLandmarks(src)
	if err != nil {
		return nil, errors.Wrap(err, "error extracting landmarks from src")
	}

	if len(srcLandmarks) == 0 {
		return nil, errors.New("no face detected in src image")
	}

	destBlurred := imaging.Blur(dest, blur)
	destBlurredAligned := image.NewRGBA(dest.Bounds())
	draw.Draw(destBlurredAligned, dest.Bounds(), destBlurred, image.ZP, draw.Src)
	destLandmarks, destCrops, err := extractor.ExtractLandmarks(destBlurredAligned)
	if err != nil {
		return nil, errors.Wrap(err, "error extracting landmarks from dest")
	}

	for i := range destLandmarks {
		fmt.Println("bounds before", destCrops[i].Bounds())
		destCrops[i], err = swap(detector, srcCrops[0], destCrops[i], blur)
		if err != nil {
			return nil, errors.Wrapf(err, "error swapping face %d", i)
		}
		fmt.Println("bounds after", destCrops[i].Bounds())

		// gg.SavePNG(fmt.Sprintf("out-swap-%d.png", i), destCrops[i])
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

func swap(detector LandmarkDetector, src, dest image.Image, blur float64) (image.Image, error) {
	// gg.SavePNG(fmt.Sprintf("out-swap-crop-src-%d.png", counter), src)
	// gg.SavePNG(fmt.Sprintf("out-swap-crop-dest-%d.png", counter), dest)

	destBlurred := imaging.Blur(dest, blur)
	destBlurredAligned := image.NewRGBA(dest.Bounds())
	draw.Draw(destBlurredAligned, dest.Bounds(), destBlurred, image.ZP, draw.Src)

	src = resize.Resize(uint(dest.Bounds().Dx()), uint(dest.Bounds().Dy()), src, resize.NearestNeighbor)

	srcLM, err := detector.Detect(src)
	if err != nil {
		return nil, err
	}
	srcLandmarks := srcLM.PointsOnImage(src)
	destLM, err := detector.Detect(destBlurredAligned)
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
		return nil, errors.Wrap(err, "error distorting face")
	}

	out := image.NewRGBA(dest.Bounds())
	draw.Draw(out, out.Bounds(), dest, dest.Bounds().Min, draw.Src)

	maskIn := image.NewRGBA(out.Bounds())
	mask := maskFromPolygon(maskIn, simplify(destLandmarks))
	maskAligned := image.NewRGBA(out.Bounds())
	draw.Draw(maskAligned, out.Bounds(), mask, image.ZP, draw.Src)

	// gg.SavePNG(fmt.Sprintf("out-swap-mask-%d.png", counter), distorted)

	distortedAligned := image.NewRGBA(out.Bounds())
	draw.Draw(distortedAligned, out.Bounds(), distorted, image.ZP, draw.Src)

	blend(distortedAligned, destBlurredAligned)
	feather(distortedAligned, maskAligned, out, srcLandmarks[33])

	fmt.Println("out bounds", out.Bounds())
	fmt.Println("dest bounds", dest.Bounds())
	fmt.Println("mask bounds", maskAligned.Bounds())
	fmt.Println("distorted bounds", distortedAligned.Bounds())
	draw.DrawMask(out, out.Bounds(), distortedAligned, out.Bounds().Min, maskAligned, out.Bounds().Min, draw.Over)

	// gg.SavePNG(fmt.Sprintf("out-swap-truc-%d.png", counter), out)
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
	var out []image.Point

	for i := 0; i < 17; i++ {
		out = append(out, landmarks[i])
	}

	for i := 26; i >= 17; i-- {
		out = append(out, moveUp(landmarks[8], landmarks[i]))
	}

	return out
}

const factor = 0.1

func moveUp(base, target image.Point) image.Point {
	target.X += int(factor * float32(target.X-base.X))
	target.Y += int(factor * float32(target.Y-base.Y))
	return target
}

type values struct {
	v []float64

	minV      float64
	minSet    bool
	maxV      float64
	maxSet    bool
	avgV      float64
	avgSet    bool
	medianV   float64
	medianSet bool
}

func (v *values) min() float64 {
	if v.minSet {
		return v.minV
	}
	min := 100000.0
	for _, vv := range v.v {
		if vv < min {
			min = vv
		}
	}
	v.minSet = true
	v.minV = min
	return min
}

func (v *values) max() float64 {
	if v.maxSet {
		return v.maxV
	}
	max := 0.0
	for _, vv := range v.v {
		if vv > max {
			max = vv
		}
	}
	v.maxSet = true
	v.maxV = max
	return max
}

func (v *values) avg() float64 {
	if v.avgSet {
		return v.avgV
	}
	sum := 0.0
	for _, vv := range v.v {
		sum += vv
	}
	avg := sum / float64(len(v.v))
	v.avgSet = true
	v.avgV = avg
	return avg
}

func (v *values) median() float64 {
	if v.medianSet {
		return v.medianV
	}

	sort.Slice(v.v, func(i, j int) bool { return v.v[i] < v.v[j] })

	median := v.v[len(v.v)/2]
	v.medianSet = true
	v.medianV = median
	return median
}

func blend(on *image.RGBA, to image.Image) {
	// gg.SavePNG("out-blurred.png", to)
	var onH, onS, onL, toH, toS, toL values
	for x := to.Bounds().Min.X; x < to.Bounds().Max.X; x++ {
		for y := to.Bounds().Min.Y; y < to.Bounds().Max.Y; y++ {
			onColor, visible := colorful.MakeColor(on.At(x, y))
			if !visible {
				continue
			}
			toColor, visible := colorful.MakeColor(to.At(x, y))
			if !visible {
				continue
			}

			h, s, l := onColor.Hsl()
			onH.v = append(onH.v, h)
			onS.v = append(onS.v, s)
			onL.v = append(onL.v, l)

			h, s, l = toColor.Hsl()
			toH.v = append(toH.v, h)
			toS.v = append(toS.v, s)
			toL.v = append(toL.v, l)
		}
	}

	for x := on.Bounds().Min.X; x < on.Bounds().Max.X; x++ {
		for y := on.Bounds().Min.Y; y < on.Bounds().Max.Y; y++ {
			onColor, visible := colorful.MakeColor(on.At(x, y))
			if !visible {
				continue
			}

			h, s, l := onColor.Hsl()
			if h < onH.median() {
				h = (h-onH.min())*(toH.median()-toH.min())/(onH.median()-onH.min()) + toH.min()
			} else {
				h = (h-onH.median())*(toH.max()-toH.median())/(onH.max()-onH.median()) + toH.median()
			}
			s = toS.median()
			if l < onL.median() {
				l = (l-onL.min())*(toL.median()-toL.min())/(onL.median()-onL.min()) + toL.min()
			} else {
				l = (l-onL.median())*(toL.max()-toL.median())/(onL.max()-onL.median()) + toL.median()
			}
			on.Set(x, y, colorful.Hsl(h, s, l))
		}
	}
}

func feather(on, mask, to *image.RGBA, center image.Point) {
	center.X += on.Bounds().Min.X
	center.Y += on.Bounds().Min.Y

	maxDist := 0.0
	for x := on.Bounds().Min.X; x < on.Bounds().Max.X; x++ {
		for y := on.Bounds().Min.Y; y < on.Bounds().Max.Y; y++ {
			if _, _, _, a := mask.At(x, y).RGBA(); a == 65535 {
				continue
			}

			distToCenter := float64((center.X-x)*(center.X-x) + (center.Y-y)*(center.Y-y))
			if distToCenter > maxDist {
				maxDist = distToCenter
			}
		}
	}

	for x := on.Bounds().Min.X; x < on.Bounds().Max.X; x++ {
		for y := on.Bounds().Min.Y; y < on.Bounds().Max.Y; y++ {
			onColor, visible := colorful.MakeColor(on.At(x, y))
			if !visible {
				continue
			}
			toColor, visible := colorful.MakeColor(to.At(x, y))
			if !visible {
				continue
			}

			distToCenter := float64((center.X-x)*(center.X-x) + (center.Y-y)*(center.Y-y))
			if distToCenter > maxDist {
				continue
			}

			t := math.Pow(distToCenter/maxDist+0.6, 4)
			if t > 1 {
				t = 1
			}
			on.Set(x, y, onColor.BlendLab(toColor, t))
		}
	}
}
