package knn

import (
	"container/heap"
	"errors"
	"fmt"
)

func L2nns(qy []float64, db [][]float64, k int, opts ...L2nnsOptions) ([]int, []float64, error) {
	if qy == nil || db == nil {
		return nil, nil, errors.New("input slices must not be nil")
	}
	if len(qy) == 0 || len(db) == 0 {
		return nil, nil, errors.New("input slices must not be empty")
	}
	if k <= 0 {
		return nil, nil, errors.New("k must be a positive integer")
	}

	recall_target := 0.95
	if len(opts) > 0 {
		recall_target = opts[0].RecallTarget
	}
	if !(0 < recall_target && recall_target <= 1) {
		return nil, nil, fmt.Errorf("recall_target must be between 0 and 1")
	}

	fmt.Printf("L2nns: qy=%d, db=%d:%d, k=%d, rt=%f\n", len(qy), len(db), len(db[0]), k, recall_target)

	scores, err := Einsum(qy, db)
	if err != nil {
		return nil, nil, err
	}

	db_halfnorm := HalfNorm(db)

	dists := make([]float64, len(scores))
	for i := range scores {
		dists[i] = db_halfnorm[i] - scores[i]
	}

	return approxMinK(dists, k, recall_target)
}

func approxMinK(dists []float64, k int, recallTarget float64) ([]int, []float64, error) {
	if dists == nil {
		return nil, nil, errors.New("input slice must not be nil")
	}
	if k > len(dists) {
		fmt.Printf("k is greater than dist, setting k to length dist: k=%d, len(dists)=%d ...\n", k, len(dists))
		k = len(dists)
	}

	n_samples := int(float64(k) / recallTarget)
	if n_samples > len(dists) {
		n_samples = len(dists)
	}

	sampled_dists := make([]float64, n_samples)
	for i := range sampled_dists {
		sampled_dists[i] = dists[i]
	}

	h := &MinHeap{}
	heap.Init(h)

	for i, v := range sampled_dists {
		heap.Push(h, Result{Index: i, Distance: v})
		if h.Len() > k {
			heap.Pop(h)
		}
	}

	indices := make([]int, k)
	values := make([]float64, k)
	for i := 0; i < k; i++ {
		result := heap.Pop(h).(Result)
		indices[i] = result.Index
		values[i] = result.Distance
	}

	return indices, values, nil
}
