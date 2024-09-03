package knn

import (
	"runtime"
	"sync"
)

func (s *Search[T]) Manhattan(i *int) T {
	var sum T
	for j := 0; j < len(s.Query.Values.Vector); j++ {
		sum += Abs(s.Query.Values.Vector[j] - s.Data.Values.Matrix[*i][j])
	}
	return sum
}

func (s *Search[T]) Einsum() []T {
	qCols := s.Query.Shape[0]
	dRows := s.Data.Shape[0]
	result := make([]T, dRows)

	if s.Multithread {
		if s.MaxWorkers == 0 {
			s.MaxWorkers = runtime.NumCPU()
		}

		var wg sync.WaitGroup
		sem := make(chan struct{}, s.MaxWorkers)
		var mu sync.Mutex

		worker := func(s *Search[T], i int, wg *sync.WaitGroup) {
			defer wg.Done()
			dot := T(0)
			for j := 0; j < qCols; j++ {
				dot += s.Query.Values.Vector[j] * s.Data.Values.Matrix[i][j]
			}

			mu.Lock()
			result[i] = dot
			mu.Unlock()

			<-sem
		}

		for i := 0; i < dRows; i++ {
			sem <- struct{}{}
			wg.Add(1)
			go worker(s, i, &wg)
		}

		wg.Wait()
		close(sem)

		return result
	}

	for i := 0; i < dRows; i++ {
		dot := T(0)
		for j := 0; j < qCols; j++ {
			dot += s.Query.Values.Vector[j] * s.Data.Values.Matrix[i][j]
		}
		result[i] = dot
	}

	return result
}

func (s *Search[T]) HalfNorm() []T {
	dRows := s.Data.Shape[0]
	dCols := s.Data.Shape[1]
	result := make([]T, dRows)

	if s.Multithread {
		if s.MaxWorkers == 0 {
			s.MaxWorkers = runtime.NumCPU()
		}

		var wg sync.WaitGroup
		sem := make(chan struct{}, s.MaxWorkers)
		var mu sync.Mutex

		worker := func(s *Search[T], i int, wg *sync.WaitGroup) {
			defer wg.Done()
			halfnorm := T(0)
			for j := 0; j < dCols; j++ {
				halfnorm += s.Data.Values.Matrix[i][j] * s.Data.Values.Matrix[i][j]
			}

			mu.Lock()
			result[i] = halfnorm * 0.5
			mu.Unlock()

			<-sem
		}

		for i := 0; i < dRows; i++ {
			sem <- struct{}{}
			wg.Add(1)
			go worker(s, i, &wg)
		}

		wg.Wait()
		close(sem)

		return result
	}

	for i := 0; i < dRows; i++ {
		halfnorm := T(0)
		for j := 0; j < dCols; j++ {
			halfnorm += s.Data.Values.Matrix[i][j] * s.Data.Values.Matrix[i][j]
		}
		result[i] = halfnorm * 0.5
	}

	return result
}

func (s *Search[T]) EstimateBinSize() int {
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
		if uint64(s.Data.Shape[0]) < binSize.threshold {
			return int(binSize.value)
		}
	}
	return 64
}

func Abs[T float32 | float64](a T) T {
	if a < 0 {
		return -a
	}
	return a
}