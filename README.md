# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

This library is a simple KNN Search for embeddings.
* L1Distance(Manhattan)
* L2Distance(Euclidean)
* MIPS (Maximum Inner Product Search)

---

This library is intended to be lightweight and quick KNN search for reasonably sized datasets. e.g small business e-comm, blogs, etc...

For a more advanced implementation, see libraries:
* FAISS - https://github.com/facebookresearch/faiss
* Annoy - https://github.com/spotify/annoy

## Installation
```sh
go get github.com/carter4299/go-knn
```

## Functions

#### MIPS
```go
type BinSize struct {
	Value int
}

func MIPSnns(qy []float64, db [][]float64, k int, opts ...BinSize) ([]int, []float64, error)
```

---

#### L2

```go 
type RecallTarget struct {
	Value float64
}
func L2nns(qy []float64, db [][]float64, k int, opts ...RecallTarget) ([]int, []float64, error)
```

---

#### L1
```go 
func L1nns(qy []float64, db [][]float64, k int) ([]int, []float64, error)
```

## Usage
```go
package main

import (
	"fmt"
	knn "github.com/carter4299/go-knn"
)
func main() {
	database := [][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		{0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
		{0.7, 0.8, 0.9, 0.1, 0.2, 0.3},
	}
	query := []float64{0.2, 0.3, 0.4, 0.5, 0.6, 0.7}

	s := &knn.New{
		Data: database,
		K:    2,
	}

	indices, values, err := s.Search(query, knn.L2)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Nearerst neighbor:", database[indices[0]])
	fmt.Println("Indices:", indices)
	fmt.Println("Values:", values)
}
```

## Example using OpenAI Ada
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
		"The artist enjoys painting in the studio.",
		"The musician enjoys playing music on the stage.",
		"The fisherman enjoys fishing in the river.",
		"The gardener enjoys planting flowers in the garden.",
		"The writer enjoys writing books in the library.",
		"The programmer enjoys coding software in the office.",
		"The engineer enjoys designing machines in the factory.",
		"The mechanic enjoys fixing cars in the garage.",
		"The electrician enjoys fixing wires in the house.",
		"The plumber enjoys fixing pipes in the bathroom.",
	}

	query_sentence := "I am a scientist who enjoys fishing when I'm not in the lab."

	var query []float64
	var database [][]float64

	for _, sentence := range sentences {
		queryReq := openai.EmbeddingRequest{
			Input: []string{sentence},
			Model: openai.AdaEmbeddingV2,
		}

		queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
		if err != nil {
			log.Fatal("Error creating query embedding:", err)
		}

		database = append(database, float32ToFloat64(queryResponse.Data[0].Embedding))
	}

	queryReq := openai.EmbeddingRequest{
		Input: []string{query_sentence},
		Model: openai.AdaEmbeddingV2,
	}

	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		log.Fatal("Error creating target embedding:", err)
	}

	query = float32ToFloat64(queryResponse.Data[0].Embedding)

	s := &knn.New{
		Data: database,
		K:    3,
	}

	indices, values, err := s.Search(query, "MIPS", knn.BinSize{Value: 2})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Query Sentence:", query_sentence)
	fmt.Println("Nearerst neighbor[0]:", sentences[indices[0]])
	fmt.Println("Nearerst neighbor[1]:", sentences[indices[1]])
	fmt.Println("Indices:", indices)
	fmt.Println("Values:", values)
}
```
Output:
```
MIPS: qy=1536, db=17:1536, k=3, bs=2
Query Sentence: I am a scientist who enjoys fishing when I'm not in the lab.
Nearerst neighbor[0]: The scientist enjoys conducting experiments in the laboratory.
Nearerst neighbor[1]: The fisherman enjoys fishing in the river.
Indices: [5 9 0]
Values: [0.8774271924221423 0.8690259037780471 0.828193179347956]
```

## Sources:
**[TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/abs/2206.14286)**