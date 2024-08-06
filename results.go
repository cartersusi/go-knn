package knn

type MinHeap []Result

type Result struct {
	Index    int
	Distance float64
}

type MaxHeap struct {
	results []Result
}

func (h *MaxHeap) Len() int           { return len(h.results) }
func (h *MaxHeap) Less(i, j int) bool { return h.results[i].Distance > h.results[j].Distance }
func (h *MaxHeap) Swap(i, j int)      { h.results[i], h.results[j] = h.results[j], h.results[i] }
func (h *MaxHeap) Push(x interface{}) { h.results = append(h.results, x.(Result)) }
func (h *MaxHeap) Pop() interface{} {
	x := h.results[len(h.results)-1]
	h.results = h.results[:len(h.results)-1]
	return x
}
func (h *MaxHeap) Peek() interface{} { return h.results[0] }
