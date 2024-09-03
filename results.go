package knn

import (
	"container/heap"
)

type Result[T float32 | float64] struct {
	Index    int
	Distance T
}

type MaxHeap[T float32 | float64] struct {
	results []Result[T]
}

func (h *MaxHeap[T]) Len() int           { return len(h.results) }
func (h *MaxHeap[T]) Less(i, j int) bool { return h.results[i].Distance > h.results[j].Distance }
func (h *MaxHeap[T]) Swap(i, j int)      { h.results[i], h.results[j] = h.results[j], h.results[i] }
func (h *MaxHeap[T]) Push(x interface{}) {
	h.results = append(h.results, x.(Result[T]))
}
func (h *MaxHeap[T]) Pop() interface{} {
	x := h.results[len(h.results)-1]
	h.results = h.results[:len(h.results)-1]
	return x
}
func (h *MaxHeap[T]) Peek() interface{} {
	return h.results[0]
}

// maybe move ??
func (h *MaxHeap[T]) Process(i *int, k *int, distance *T) {
	if h.Len() < *k {
		heap.Push(h, Result[T]{Index: *i, Distance: *distance})
	} else if *distance < h.Peek().(Result[T]).Distance {
		heap.Pop(h)
		heap.Push(h, Result[T]{Index: *i, Distance: *distance})
	}
}
