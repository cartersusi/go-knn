# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

This library is a simple KNN Search for embeddings.\
Current distance function support:
* L1Distance(Manhattan)
* L2Distance(Euclidean)

---

This library is intended to be lightweight and quick KNN search with "good enough" accuracy for reasonably sized datasets. e.g small business e-comm, blogs, etc...

For a more advanced implementation, see libraries:
* FAISS - https://github.com/facebookresearch/faiss
* Annoy - https://github.com/spotify/annoy

## Installation
```sh
go get github.com/carter4299/go-knn
```

## Usage
```go
package main

import (
	"fmt"

	knn "github.com/carter4299/go-knn"
)

func main() {
	embeddings := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}
	target := []float64{0.2, 0.3, 0.4}

	knn, err := knn.NewKNN(len(target), embeddings, target, knn.L1Distance)
	if err != nil {
		fmt.Println(err)
		return
	}

	indices, err := knn.Search()
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Nearerst neighbor:", embeddings[indices[0]])
	fmt.Println("Indices of k nearest neighbors:", indices)
}
```

## Full Example using OpenAI Ada
```go
package main

import (
	"context"
	"fmt"
	"log"

	knn "github.com/carter4299/go-knn"
	openai "github.com/sashabaranov/go-openai"
)

func float32ToFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func main() {
	openai_token := "my-key"
	client := openai.NewClient(openai_token)

	sentences := []string{
		"The sailor enjoys sailing on a boat in the sea.",
		"The carpenter enjoys building houses with wood.",
		"The athlete enjoys running on the track.",
		"The chef enjoys cooking in the kitchen.",
		"The doctor enjoys helping patients in the hospital.",
		"The scientist enjoys conducting experiments in the laboratory.",
		"The teacher enjoys teaching students in the classroom.",
	}

	target_sentence := "I am a fisherman who enjoys fishing on my boat."

	var target []float64
	var data [][]float64

	for _, sentence := range sentences {
		queryReq := openai.EmbeddingRequest{
			Input: []string{sentence},
			Model: openai.AdaEmbeddingV2,
		}

		queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
		if err != nil {
			log.Fatal("Error creating query embedding:", err)
		}

		data = append(data, float32ToFloat64(queryResponse.Data[0].Embedding))
	}

	targetReq := openai.EmbeddingRequest{
		Input: []string{target_sentence},
		Model: openai.AdaEmbeddingV2,
	}

	targetResponse, err := client.CreateEmbeddings(context.Background(), targetReq)
	if err != nil {
		log.Fatal("Error creating target embedding:", err)
	}

	target = float32ToFloat64(targetResponse.Data[0].Embedding)

	knn, err := knn.NewKNN(len(target), data, target, knn.L1Distance) // Use L2Distance for Euclidean distance
	if err != nil {
		fmt.Println(err)
		return
	}

	indices, err := knn.Search()
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Nearerst neighbor:", sentences[indices[0]])
	fmt.Println("Indices of k nearest neighbors:", indices)
}
```
Output:
```
Nearerst neighbor: The sailor enjoys sailing on a boat in the sea.
Indices of k nearest neighbors: [0 3 1 5 4 2 6]
```
