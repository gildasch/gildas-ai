package api

import (
	"fmt"
	"math"
	"testing"
)

type mockCluser struct {
	distances [][]float32
}

func (m *mockCluser) Len() int                  { return len(m.distances) }
func (m *mockCluser) Distance(i, j int) float32 { return m.distances[i][j] }

func TestProject2D(t *testing.T) {
	c := &mockCluser{
		distances: [][]float32{
			[]float32{0, 4, 3, 6},
			[]float32{4, 0, 1, 9},
			[]float32{3, 1, 0, 7},
			[]float32{6, 9, 7, 0},
		},
	}

	points := project2D(c)

	fmt.Println(points[0].X)
	fmt.Println(points)

	for i := 0; i < c.Len(); i++ {
		for j := i + 1; j < c.Len(); j++ {
			fmt.Printf("%d to %d: %f\n", i, j,
				math.Sqrt(float64((points[i].X-points[j].X)*(points[i].X-points[j].X)+
					(points[i].Y-points[j].Y)*(points[i].Y-points[j].Y))))
		}
	}
}

func TestMove(t *testing.T) {
	fmt.Println(move(point{X: 0, Y: 2}, point{X: 2, Y: 0}, 4))
	fmt.Println(move(point{X: 0, Y: 2}, point{X: 2, Y: 0}, 1))
}
