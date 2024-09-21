package knn

import (
	"container/heap"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

type Search[T float32 | float64] struct {
	Data        *Tensor[T]
	Query       *Tensor[T]
	Multithread bool
	MaxWorkers  int
	SIMD        bool
}

type Neighbors[T any] struct {
	Indices []int
	Values  []T
}

const (
	L1 = iota
	L2
	MIPS
)

func (s *Search[T]) L1(k int) (Neighbors[T], error) {
	if err := s.checker(k); err != nil {
		return Neighbors[T]{}, err
	}

	h := &MaxHeap[T]{}
	heap.Init(h)

	if s.Multithread {
		if s.MaxWorkers == 0 {
			s.MaxWorkers = runtime.NumCPU()
		}
		n_tasks := len(s.Data.Values.([][]T))

		var wg sync.WaitGroup
		results := make(chan struct {
			index    int
			distance T
		}, n_tasks)

		worker := func(s *Search[T], i int) {
			defer wg.Done()
			distance := s.Manhattan(&i)
			results <- struct {
				index    int
				distance T
			}{i, distance}
		}

		for i := 0; i < n_tasks; i++ {
			wg.Add(1)
			go worker(s, i)
		}

		go func() {
			wg.Wait()
			close(results)
		}()

		for result := range results {
			h.Process(&result.index, &k, &result.distance)
		}

		return s.ret(&k, h)
	}

	for i := 0; i < len(s.Data.Values.([][]T)); i++ {
		distance := s.Manhattan(&i)
		h.Process(&i, &k, &distance)
	}

	return s.ret(&k, h)
}

func (s *Search[T]) L2(k int) (Neighbors[T], error) {
	if err := s.checker(k); err != nil {
		return Neighbors[T]{}, err
	}

	h := &MaxHeap[T]{}
	heap.Init(h)

	dots := s.Einsum()
	halfnorm := s.HalfNorm()

	for i := range dots {
		distance := halfnorm[i] - dots[i]
		h.Process(&i, &k, &distance)
	}

	return s.ret(&k, h)
}

func (s *Search[T]) MIPS(k int, opts ...interface{}) (Neighbors[T], error) {
	if err := s.checker(k); err != nil {
		return Neighbors[T]{}, err
	}

	if s.Multithread {
		Log("MIPS does not support multithreading for now", Warning)
	}

	bs := s.EstimateBinSize()
	if len(opts) > 0 {
		var ok bool
		bs, ok = opts[0].(int)
		if !ok {
			return Neighbors[T]{}, errors.New("invalid options for MIPS")
		}
	}
	if bs <= 0 || bs > 64 || bs > s.Query.Shape[0] {
		return Neighbors[T]{}, errors.New("invalid bin_size")
	}

	// Warnings: just an observation, magic number ig
	if (s.Data.Shape[0]/bs) < 8 && bs > 1 {
		Log("bin_size is too large for the size of the database. This may lead to unexpected results.", Warning)
	}

	bin_sizes := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 32: true, 64: true}
	if !bin_sizes[bs] {
		Log("bin_size is not a power of 2. This may lead to unexpected results.", Warning)
	}

	scores := s.Einsum()
	if scores == nil {
		return Neighbors[T]{}, errors.New("unknown error while calculating scores")
	}
	if k > len(scores) {
		return Neighbors[T]{}, errors.New("k must be less than the length of the scores vector")
	}

	// gets messy, see https://arxiv.org/pdf/2206.14286

	N := len(scores)
	L := N / bs

	if N%bs != 0 {
		L++
	}

	V := make([]T, L)
	A := make([]int, L)

	jb := 16
	for jj := 0; jj < N-jb+1; jj += jb {
		yi := make([]T, jb)
		for j := jj; j < jj+jb; j++ {
			yi[j-jj] = scores[j]
		}
		for j := jj; j < jj+jb; j++ {
			l := (j >> uint(bs))
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
		l := (jj >> uint(bs))
		b := yi > V[l]
		if b {
			V[l] = yi
			A[l] = jj
		}
	}

	indices := make([]int, k)
	values := make([]T, k)
	for i := 0; i < k; i++ {
		maxValue := T(-1e9)
		maxIndex := -1
		for j := 0; j < N; j++ {
			if scores[j] > maxValue {
				maxValue = scores[j]
				maxIndex = j
			}
		}
		indices[i] = maxIndex
		values[i] = maxValue
		scores[maxIndex] = T(-1e9) // Mark this score as used
	}

	return Neighbors[T]{Values: values, Indices: indices}, nil
}

func (s *Search[T]) checker(k int) error {
	if k <= 0 || k > len(s.Data.Values.([][]T)) {
		return errors.New("k must be greater than 0 and less than the length of the data")
	}
	if s.Data == nil || s.Query == nil {
		return errors.New("data and query tensors must be initialized")
	}

	if s.Data.Rank != 2 || s.Query.Rank != 1 {
		return errors.New("data must be a matrix and query must be a vector")
	}

	if s.Data.Shape[1] != s.Query.Shape[0] {
		return errors.New("data and query dimensions do not match")
	}

	return nil
}

func (s *Search[T]) ret(k *int, h *MaxHeap[T]) (Neighbors[T], error) {
	indices := make([]int, *k)
	values := make([]T, *k)
	for i := 0; i < *k; i++ {
		result := heap.Pop(h).(Result[T])
		indices[*k-1-i] = result.Index
		values[*k-1-i] = result.Distance
	}

	return Neighbors[T]{Values: values, Indices: indices}, nil
}

func (s *Search[T]) PrintDistances() {
	fmt.Println("Available options for Search:")
	fmt.Println("\t1. L1(k int)")
	fmt.Println("\t2. L2(k int)")
	fmt.Println("\t3. MIPS(k int, ?bin_size int)")
}

func (s *Search[T]) GetSize() T {
	size := T(0)
	if s.Data == nil || s.Query == nil {
		return size
	}
	size += T(unsafe.Sizeof(s))
	size += T(unsafe.Sizeof(s.Data.Values.([][]T)))
	for _, row := range s.Data.Values.([][]T) {
		size += T(unsafe.Sizeof(row))
		size += T(len(row)) * T(unsafe.Sizeof(T(0)))
	}
	size += T(unsafe.Sizeof(s.Data.Shape) * 2)
	size += T(unsafe.Sizeof(s.Data.Type))
	size += T(unsafe.Sizeof(s.Data.Rank))

	size += T(unsafe.Sizeof(s.Query.Values.([]T)))
	size += T(len(s.Query.Values.([]T))) * T(unsafe.Sizeof(T(0)))
	size += T(unsafe.Sizeof(s.Query.Shape))
	size += T(unsafe.Sizeof(s.Query.Type))
	size += T(unsafe.Sizeof(s.Query.Rank))

	size += T(unsafe.Sizeof(s.Multithread))
	size += T(unsafe.Sizeof(s.MaxWorkers))

	return size / (1024 * 1024)
}

func (s *Search[T]) PrintTypes() {
	fmt.Println(`Search[T float32 | float64] {
  Data: *Tensor[T],
  Query: *Tensor[T],
  Multithread: bool,
  MaxWorkers: int,
}`)
}

func (s *Search[T]) Print() {
	fmt.Printf("Data Tensor:\n  Shape: %v\n  Rank: %d\n  Type: %v\n", s.Data.Shape, s.Data.Rank, s.Data.Type)
	fmt.Printf("Query Tensor:\n  Shape: %v\n  Rank: %d\n  Type: %v\n", s.Query.Shape, s.Query.Rank, s.Query.Type)
	fmt.Printf("Multithread: %v\nMaxWorkers: %d\n", s.Multithread, s.MaxWorkers)
	fmt.Printf("Total Size: %f MB\n\n", s.GetSize())
}
