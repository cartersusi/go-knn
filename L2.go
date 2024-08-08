package knn

import (
	"container/heap"
	"errors"
	"fmt"
)

func L2nns(qy []float64, db [][]float64, k int, opts ...RecallTarget) ([]int, []float64, error) {
	err := Validate(qy, db, k)
	if err != nil {
		return nil, nil, err
	}

	recall_target := 0.95
	if len(opts) > 0 {
		recall_target = opts[0].Value
	}
	if !(0 < recall_target && recall_target <= 1) {
		return nil, nil, fmt.Errorf("recall_target must be between 0 and 1")
	}

	Log(fmt.Sprintf("L2: qy=%d, db=%d:%d, k=%d, rt=%f\n", len(qy), len(db), len(db[0]), k, recall_target), Info)

	dots, err := Einsum(qy, db)
	if err != nil {
		return nil, nil, err
	}

	db_halfnorm := HalfNorm(db)

	dists := make([]float64, len(dots))
	for i := range dots {
		dists[i] = db_halfnorm[i] - dots[i]
	}

	return approxMinK(dists, k, recall_target)
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
