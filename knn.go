package knn

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
)

type KNN struct {
	k        int
	data     [][]float64
	target   []float64
	distance func([]float64, []float64) (float64, error)
}

type Result struct {
	Index    int
	Distance float64
}

type MinHeap []Result

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Result))
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func check_slices(a, b []float64) error {
	if a == nil || b == nil {
		return errors.New("input slices must not be nil")
	}
	if len(a) != len(b) {
		return errors.New("input slices must have the same length")
	}
	return nil
}

func NewKNN(k int, data [][]float64, target []float64, distance func([]float64, []float64) (float64, error)) (*KNN, error) {
	if k <= 0 {
		return nil, errors.New("k must be a positive integer")
	}
	if len(data) == 0 {
		return nil, errors.New("data must not be empty")
	}
	if len(target) == 0 {
		return nil, errors.New("target must not be empty")
	}
	if distance == nil {
		return nil, errors.New("distance function must not be nil")
	}

	fmt.Printf("NewKNN: k=%d, data=%d:%d, target=%d\n", k, len(data), len(data[0]), len(target))
	return &KNN{
		k:        k,
		data:     data,
		target:   target,
		distance: distance,
	}, nil
}

func L1Distance(a, b []float64) (float64, error) {
	err := check_slices(a, b)
	if err != nil {
		return 0, err
	}

	var result float64
	for i := 0; i < len(a); i++ {
		result += math.Abs(a[i] - b[i])
	}

	return result, nil
}

func L2Distance(a, b []float64) (float64, error) {
	err := check_slices(a, b)
	if err != nil {
		return 0, err
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Pow(a[i]-b[i], 2)
	}
	return math.Sqrt(sum), nil
}

func (knn *KNN) Search() ([]int, error) {
	h := &MinHeap{}
	heap.Init(h)

	for i, dataPoint := range knn.data {
		distance, err := knn.distance(dataPoint, knn.target)
		if err != nil {
			return nil, err
		}

		if h.Len() < knn.k {
			heap.Push(h, Result{Index: i, Distance: distance})
		} else if distance < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, Result{Index: i, Distance: distance})
		}
	}

	var indices []int
	for h.Len() > 0 {
		result := heap.Pop(h).(Result)
		indices = append(indices, result.Index)
	}

	for i, j := 0, len(indices)-1; i < j; i, j = i+1, j-1 {
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices, nil
}
