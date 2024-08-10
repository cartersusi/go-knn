package knn

import (
	"container/heap"
	"fmt"
)

func L1nns(qy Tensor, db Tensor, k int) ([]int, []float64, error) {
	Log(fmt.Sprintf("L1: qy=%v, db=%v, k=%d", qy.Shape, db.Shape, k), Info)

	h := &MaxHeap{}
	heap.Init(h)

	process := func(i int, distance float64) {
		if h.Len() < k {
			heap.Push(h, Result{Index: i, Distance: distance})
		} else if distance < h.Peek().(Result).Distance {
			heap.Pop(h)
			heap.Push(h, Result{Index: i, Distance: distance})
		}
	}

	switch qy.Type {
	case Double:
		qyv := qy.Values.([]float64)
		for i, dp := range db.Values.([][]float64) {
			distance := Manhattanf64(qyv, dp)
			process(i, distance)
		}
	case Float:
		qyv := qy.Values.([]float32)
		for i, dp := range db.Values.([][]float32) {
			distance := Manhattanf32(qyv, dp)
			process(i, float64(distance))
		}
	default:
		return nil, nil, fmt.Errorf("unsupported type %v", db.Type)
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
