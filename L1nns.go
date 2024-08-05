package knn

import (
	"container/heap"
	"errors"
	"fmt"
)

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

	for i, dp := range db {
		distance, err := Euclidean(dp, qy)
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
