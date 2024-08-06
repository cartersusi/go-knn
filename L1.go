package knn

import (
	"container/heap"
	"fmt"
)

func L1nns(qy []float64, db [][]float64, k int) ([]int, []float64, error) {
	err := Validate(qy, db, k)
	if err != nil {
		return nil, nil, err
	}

	fmt.Printf("L1: qy=%d, db=%d:%d, k=%d\n", len(qy), len(db), len(db[0]), k)

	h := &MaxHeap{}
	heap.Init(h)

	for i, dp := range db {
		distance := Manhattan(dp, qy)

		if h.Len() < k {
			heap.Push(h, Result{Index: i, Distance: distance})
		} else if distance < h.Peek().(Result).Distance {
			heap.Pop(h)
			heap.Push(h, Result{Index: i, Distance: distance})
		}
	}

	indices := make([]int, k)
	values := make([]float64, k)
	for i := 0; i < k; i++ {
		result := heap.Pop(h).(Result)
		indices[k-1-i] = result.Index
		values[k-1-i] = result.Distance
	}

	return indices, values, nil
}
