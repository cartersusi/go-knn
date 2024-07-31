package knn

import (
	"container/heap"
	"errors"
	"math"
	"runtime"
	"sync"
)

type KNN struct {
	k        int
	data     [][]float64
	target   []float64
	distance func([]float64, []float64) (float64, error)
}

func NewKNN(k int, data [][]float64, target []float64, distance func([]float64, []float64) (float64, error)) (*KNN, error) {
	if k <= 0 {
		return nil, errors.New("k must be a positive integer")
	}
	if k != len(target) || k != len(data[0]) {
		return nil, errors.New("k must be equal to the length of the target and data")
	}
	if len(data) == 0 {
		return nil, errors.New("data must not be empty")
	}
	if len(target) == 0 {
		return nil, errors.New("target must not be empty")
	}
	if distance == nil {
		return nil, errors.New("distance function must not be nil")
	}

	for i := 0; i < len(data); i++ {
		if len(data[i]) != len(target) {
			return nil, errors.New("data and target must have the same length")
		}
	}

	return &KNN{
		k:        k,
		data:     data,
		target:   target,
		distance: distance,
	}, nil
}

type Result struct {
	Index    int
	Distance int
}

type ResultHeap []Result

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *ResultHeap) Push(x interface{}) {
	*h = append(*h, x.(Result))
}

func (h *ResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func L1Distance(a, b []float64) (float64, error) {
	if a == nil || b == nil {
		return 0, errors.New("input slices must not be nil")
	}
	numChunks := runtime.NumCPU()
	chunkSize := len(a) / numChunks

	var wg sync.WaitGroup
	wg.Add(numChunks)

	var mu sync.Mutex
	var result float64

	chunkCalc := func(start, end int) {
		defer wg.Done()
		var chunkSum float64
		for i := start; i < end; i++ {
			chunkSum += math.Abs(a[i] - b[i])
		}
		mu.Lock()
		result += chunkSum
		mu.Unlock()
	}

	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := (i + 1) * chunkSize
		if i == numChunks-1 {
			end = len(a)
		}
		go chunkCalc(start, end)
	}

	wg.Wait()

	return result, nil
}

func (knn *KNN) Search() ([]int, error) {
	results := make(chan Result)
	var wg sync.WaitGroup

	errors := make(chan error)
	var searchErr error

	for i, dataPoint := range knn.data {
		wg.Add(1)

		go func(i int, dataPoint []float64) {
			defer wg.Done()
			distance, err := knn.distance(dataPoint, knn.target)
			if err != nil {
				errors <- err
				return
			}

			results <- Result{Index: i, Distance: int(distance)}
		}(i, dataPoint)
	}

	h := &ResultHeap{}
	heap.Init(h)

	go func() {
		for result := range results {
			if h.Len() < knn.k {
				heap.Push(h, result)
			} else if result.Distance < (*h)[0].Distance {
				heap.Pop(h)
				heap.Push(h, result)
			}
		}
	}()

	go func() {
		for err := range errors {
			searchErr = err
		}
	}()

	wg.Wait()
	close(results)
	close(errors)

	if searchErr != nil {
		return nil, searchErr
	}

	var topK []int
	for h.Len() > 0 {
		result := heap.Pop(h).(Result)
		topK = append(topK, result.Index)
	}

	return topK, nil
}
