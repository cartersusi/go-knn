package knn

type Result struct {
	Index    int
	Distance float64
}

type MinHeap []Result

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Result))
}

func (h *MinHeap) Pop() interface{} {
	tmp := *h
	n := len(tmp)
	x := tmp[n-1]
	*h = tmp[0 : n-1]
	return x
}
