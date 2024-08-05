# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

This library is a simple KNN Search for embeddings.\
Current distance function support:
* L1Distance(Manhattan)
* L2Distance(Euclidean)

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

#### L2nns

```go 
type L2nnsOptions struct {
	RecallTarget float64
}
func L2nns(qy []float64, db [][]float64, k int, opts ...L2nnsOptions) ([]int, []float64, error)

// Default recall_target=0.95
indices, values, err := knn.L2nns(query, database, len(query))

// Define a recall target value with:
knn.L2nns(query, database, len(query), knn.L2nnsOptions{RecallTarget: 0.90})
```

#### L1nns
```go 
func L1nns(qy []float64, db [][]float64, k int) ([]int, []float64, error)

indices, values, err := knn.L1nns(query, database, len(query))
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
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}
	query := []float64{0.2, 0.3, 0.4}

	indices, values, err := knn.L1nns(query, database, len(query))
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Nearerst neighbor:", database[indices[0]])
	fmt.Println("Indices:", indices[0:3])
	fmt.Println("Values:", values[0:3])
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

	indices, values, err := knn.L2nns(query, database, len(database), knn.L2nnsOptions{RecallTarget: 0.90})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Nearerst neighbor:", sentences[indices[0]])
	fmt.Println("Indices:", indices[0:3])
	fmt.Println("Values:", values[0:3])
}
```
Output:
```
L2nns: qy=1536, db=7:1536, k=7, rt=0.900000
Nearerst neighbor: The scientist enjoys conducting experiments in the laboratory.
Indices: [5 0 3]
Values: [-0.3774270905672783 -0.3281931648621844 -0.3206193134616744]
```

## Sources:
**[TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/abs/2206.14286)**