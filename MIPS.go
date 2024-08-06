package knn

import (
	"errors"
	"fmt"
)

func MIPSnns(qy []float64, db [][]float64, k int, opts ...BinSize) ([]int, []float64, error) {
	err := Validate(qy, db, k)
	if err != nil {
		return nil, nil, err
	}

	bin_size := estimateBinSize(len(db))
	if len(opts) > 0 {
		bin_size = opts[0].Value
	}

	if bin_size <= 0 || bin_size > 64 {
		return nil, nil, errors.New("bin_size must be a positive integer less than or equal to 64")
	}
	if bin_size > len(qy) && bin_size > len(db[0]) {
		return nil, nil, errors.New("bin_size must be less than the length of the input slices")
	}

	// Warnings: just an observation, magic number ig
	if (len(db)/bin_size) < 8 && bin_size > 1 {
		fmt.Println("Warning: bin_size is relatively large for the input data. Information Loss and Unexpected Results may occur.")
	}

	bin_sizes := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 32: true, 64: true}
	if !bin_sizes[bin_size] {
		fmt.Println("Warning: bin_size is not a power of 2. Unexpected Results may occur.")
	}

	fmt.Printf("MIPS: qy=%d, db=%d:%d, k=%d, bs=%d\n", len(qy), len(db), len(db[0]), k, bin_size)

	scores, err := Einsum(qy, db)
	if err != nil {
		return nil, nil, err
	}

	return approxMaxK(scores, k, bin_size)
}

// https://arxiv.org/pdf/2206.14286
func approxMaxK(scores []float64, k int, binSize int) ([]int, []float64, error) {
	if scores == nil {
		return nil, nil, errors.New("unknown error while calculating scores")
	}
	if k > len(scores) {
		return nil, nil, errors.New("k must be less than the length of the scores vector")
	}

	N := len(scores)
	L := N / binSize

	if N%binSize != 0 {
		L++
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

	return topKIndices, topKValues, nil
}

func estimateBinSize(dbLen int) int {
	binSizes := []struct {
		threshold uint64
		value     uint64
	}{
		{1 << 8, 1},
		{1 << 12, 2},
		{1 << 16, 4},
		{1 << 20, 8},
		{1 << 24, 16},
		{1 << 28, 32},
	}

	for _, binSize := range binSizes {
		if uint64(dbLen) < binSize.threshold {
			return int(binSize.value)
		}
	}
	return 64
}
