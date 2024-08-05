package knn

import (
	"errors"
	"fmt"
)

type MipsOptions struct {
	BinSize int
}

func MIPS(qy []float64, db [][]float64, k int, opts ...MipsOptions) ([]int, []float64, error) {
	if qy == nil || db == nil {
		return nil, nil, errors.New("input slices must not be nil")
	}
	if len(qy) == 0 || len(db) == 0 {
		return nil, nil, errors.New("input slices must not be empty")
	}
	if k <= 0 {
		return nil, nil, errors.New("k must be a positive integer")
	}

	bin_size := 2
	if len(opts) > 0 {
		bin_size = opts[0].BinSize
	}
	if bin_size <= 0 || bin_size > 64 {
		return nil, nil, errors.New("bin_size must be a positive integer")
	}
	if bin_size > len(qy) && bin_size > len(db[0]) {
		return nil, nil, errors.New("bin_size must be less than the length of the query and db vectors")
	}

	fmt.Printf("MIPS: qy=%d, db=%d:%d, k=%d, bs=%d\n", len(qy), len(db), len(db[0]), k, bin_size)

	scores, err := Einsum(qy, db)
	if err != nil {
		return nil, nil, err
	}

	A, V := approxMaxK(scores, k, bin_size)
	return A, V, nil
}

// https://arxiv.org/pdf/2206.14286
func approxMaxK(scores []float64, k int, binSize int) ([]int, []float64) {
	N := len(scores)
	L := N / binSize

	if k > L {
		fmt.Printf("k is greater than L, setting k to L: k=%d, L=%d ...\n", k, L)
		k = L
	}

	V := make([]float64, L)
	A := make([]int, L)

	jb := 16
	for jj := 0; jj < N-jb+1; jj += jb {
		yi := make([]float64, jb)
		for j := jj; j < jj+jb; j++ {
			yi[j-jj] = scores[j]
		}
		for j := jj; j < jj+jb; j++ {
			l := (j >> uint(binSize))
			b := yi[j-jj] > V[l]
			if b {
				V[l] = yi[j-jj]
				A[l] = j
			}
		}
	}

	// Temporary fix to handle the last block, idk
	for jj := N - N%jb; jj < N; jj++ {
		yi := scores[jj]
		l := (jj >> uint(binSize))
		b := yi > V[l]
		if b {
			V[l] = yi
			A[l] = jj
		}
	}

	topKValues := make([]float64, k)
	topKIndices := make([]int, k)
	for i := 0; i < k; i++ {
		maxValue := float64(-1)
		maxIndex := -1
		for j := 0; j < L; j++ {
			if V[j] > maxValue {
				maxValue = V[j]
				maxIndex = A[j]
			}
		}
		topKValues[i] = maxValue
		topKIndices[i] = maxIndex
		V[maxIndex>>uint(binSize)] = float64(-1)
	}

	return topKIndices, topKValues
}
