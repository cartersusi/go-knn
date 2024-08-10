package knn

import (
	"container/heap"
	"errors"
	"fmt"
)

func L2nns(qy Tensor, db Tensor, k int, recall_target float64) ([]int, []float64, error) {
	if !(0 < recall_target && recall_target <= 1) {
		return nil, nil, fmt.Errorf("recall_target must be between 0 and 1")
	}

	Log(fmt.Sprintf("L2: qy=%v, db=%v, k=%d, rt=%f", qy.Shape, db.Shape, k, recall_target), Info)

	switch qy.Type {
	case Double:
		qyv := qy.Values.([]float64)
		dbv := db.Values.([][]float64)
		dists := calcDistf64(qyv, dbv)

		return approxMinK(dists, k, recall_target)
	case Float:
		qyv := qy.Values.([]float32)
		dbv := db.Values.([][]float32)
		dists := calcDistf32(qyv, dbv)

		return approxMinK(dists, k, recall_target)
	default:
		return nil, nil, fmt.Errorf("unsupported type %v", db.Type)
	}

	return nil, nil, errors.New("unknown error")
}

func calcDistf64(qy []float64, db [][]float64) []float64 {
	dots := Einsumf64(qy, db)
	db_halfnorm := HalfNormf64(db)

	dists := make([]float64, len(dots))
	for i := range dots {
		dists[i] = db_halfnorm[i] - dots[i]
	}

	return dists
}

func calcDistf32(qy []float32, db [][]float32) []float64 {
	dots := Einsumf32(qy, db)
	db_halfnorm := HalfNormf32(db)

	dists := make([]float64, len(dots))
	for i := range dots {
		dists[i] = float64(db_halfnorm[i]) - float64(dots[i])
	}

	return dists
}

func approxMinK(dists []float64, k int, recallTarget float64) ([]int, []float64, error) {
	if dists == nil {
		return nil, nil, errors.New("unknown error while calculating dists")
	}
	if k > len(dists) {
		return nil, nil, errors.New("k must be less than the length of the dists vector")
	}

	n_samples := int(float64(k) / recallTarget)
	if n_samples > len(dists) {
		n_samples = len(dists)
	}

	sampled_dists := make([]float64, n_samples)
	for i := range sampled_dists {
		sampled_dists[i] = dists[i]
	}

	h := &MaxHeap{}
	heap.Init(h)

	for i, v := range dists {
		if h.Len() < k {
			heap.Push(h, Result{Index: i, Distance: v})
		} else if v < h.Peek().(Result).Distance {
			heap.Pop(h)
			heap.Push(h, Result{Index: i, Distance: v})
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
