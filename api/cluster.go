package api

import (
	"fmt"
	"math"
	"math/rand"
)

type cluster interface {
	Len() int
	Distance(i, j int) float32
}

type point struct{ X, Y float32 }

const debug = false

func project2D(c cluster) []point {
	res := make([]point, c.Len())

	for k := 0; k < 10; k++ {
		for i := 0; i < c.Len(); i++ {
			for j := 0; j < c.Len(); j++ {
				res[i] = move(res[i], res[j], c.Distance(i, j))
			}
		}
	}

	if debug {
		for i := 0; i < c.Len(); i++ {
			for j := 0; j < c.Len(); j++ {
				actualDistance := float32(math.Sqrt(float64((res[i].X-res[j].X)*(res[i].X-res[j].X) + (res[i].Y-res[j].Y)*(res[i].Y-res[j].Y))))
				fmt.Printf("expected: %f, actual: %f\n", c.Distance(i, j), actualDistance)
			}
		}
	}

	return res
}

func move(a, b point, expectedDistance float32) point {
	actualDistance := float32(math.Sqrt(float64((a.X-b.X)*(a.X-b.X) + (a.Y-b.Y)*(a.Y-b.Y))))
	for actualDistance == 0 {
		a.X += rand.Float32()
		a.Y += rand.Float32()
		actualDistance = float32(math.Sqrt(float64((a.X-b.X)*(a.X-b.X) + (a.Y-b.Y)*(a.Y-b.Y))))
	}
	push := (actualDistance - expectedDistance) / actualDistance

	a.X += push * (b.X - a.X)
	a.Y += push * (b.Y - a.Y)

	return a
}
