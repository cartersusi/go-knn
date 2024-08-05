package knn

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
)

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
	tmp := *h
	n := len(tmp)
	x := tmp[n-1]
	*h = tmp[0 : n-1]
	return x
}

func L1nns(qy []float64, db [][]float64, k int) ([]int, []float64, error) {
	if qy == nil || db == nil {
		return nil, nil, errors.New("input slices must not be nil")
	}
	if len(qy) == 0 || len(db) == 0 {
		return nil, nil, errors.New("input slices must not be empty")
	}
	if k <= 0 {
		return nil, nil, errors.New("k must be a positive integer")
	}
	fmt.Printf("L1nns: qy=%d, db=%d:%d, k=%d\n", len(qy), len(db), len(db[0]), k)

	h := &MinHeap{}
	heap.Init(h)

	for i, dataPoint := range db {
		distance, err := L1Distance(dataPoint, qy)
		if err != nil {
			return nil, nil, err
		}

		if h.Len() < k {
			heap.Push(h, Result{Index: i, Distance: distance})
		} else if distance < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, Result{Index: i, Distance: distance})
		}
	}

	var indices []int
	var values []float64
	for h.Len() > 0 {
		result := heap.Pop(h).(Result)
		values = append(values, result.Distance)
		indices = append(indices, result.Index)
	}

	for i, j := 0, len(indices)-1; i < j; i, j = i+1, j-1 {
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices, values, nil
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