# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/cartersusi/go-knn.svg)](https://pkg.go.dev/github.com/cartersusi/go-knn)

## TODO
- SIMD

---

For more advanced GPU implementations, see:
* FAISS - https://github.com/facebookresearch/faiss
* Annoy - https://github.com/spotify/annoy

## Installation
```sh
go get github.com/cartersusi/go-knn
```

## Usage
### Importing
```go 
import "github.com/cartersusi/go-knn"
```

### Creating Tensors
Supported Scalars:
* float32
* float64

Supported Dimensions/Ranks:
* 1
* 2

**Matrix**:
```go
matrix := [][]float32{
	{0.1, 0.2, 0.3, 0.4},
	{0.4, 0.5, 0.6, 0.7},
}
m := &knn.Tensor[float32]{}
m.New(matrix)
```

**Vectors**:
```go
vector := []float32{0.2, 0.3, 0.4, 0.5}
v := &knn.Tensor[float32]{}
v.New(vector)
```

### Searching
**New Instance**
```go
s := &knn.Search{
	Data: m,		  // 2D Tensor 
	Query: v,		  // 1D Tensor
	Multithread: true,	  // Enable Multithreading (default = false), MIPS not supported
	MaxWorkers:  m.Shape[0],  // Specify MaxWorkers (default = n_cpu_cores)
}
```

**Search Options**
|fn|speed|accuracy|
|-|-|-|
|L1|1|3|
|L2|3|1|
|MIPS|2|2|
```go
nearest_neighbors, err := s.L1(2) 	// L1, k=2
nearest_neighbors, err := s.L2(1) 	// L2, k=1
nearest_neighbors, err := s.MIPS(4, 2) 	// MIPS, k=4, bin_size=2
```

**New Query**

Seach.Query stores the address of a 1D Tensor, so it can be quickly changed for a new iteration.
```go
for query := range all_queries {
	s.Query = query
	nn, _ := s.L1(2)
}
```

## Example using OpenAI Ada (MIPS)
```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/cartersusi/go-knn"
	openai "github.com/sashabaranov/go-openai"
)

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
	}

	query_sentence := "I am a scientist who enjoys fishing when I'm not in the lab."

	var vector []float32
	var matrix [][]float32

	for _, sentence := range sentences {
		queryReq := openai.EmbeddingRequest{
			Input: []string{sentence},
			Model: openai.AdaEmbeddingV2,
		}

		queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
		if err != nil {
			log.Fatal("Error creating query embedding:", err)
		}

		matrix = append(matrix, queryResponse.Data[0].Embedding)
	}

	queryReq := openai.EmbeddingRequest{
		Input: []string{query_sentence},
		Model: openai.AdaEmbeddingV2,
	}

	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		log.Fatal("Error creating target embedding:", err)
	}

	vector = queryResponse.Data[0].Embedding

	d := &knn.Tensor[float32]{}
	d.New(matrix)
	
	q := &knn.Tensor[float32]{}
	q.New(vector)

	s := &knn.Search[float32]{
		Data:  d,
		Query: q,
	}

	nn, err := s.MIPS(2, 1)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Query Sentence:", query_sentence)
	fmt.Println("Nearerst neighbor[0]:", sentences[nn.Indices[0]])
	fmt.Println("Nearerst neighbor[1]:", sentences[nn.Indices[1]])
	fmt.Println("Indices:", nn.Indices)
	fmt.Println("Values:", nn.Values)
}
```
Output:
```
Query Sentence: I am a scientist who enjoys fishing when I'm not in the lab.
Nearerst neighbor[0]: The scientist enjoys conducting experiments in the laboratory.
Nearerst neighbor[1]: The sailor enjoys sailing on a boat in the sea.
Indices: [5 0]
Values: [0.877427339553833 0.828193724155426]
```

## Sources:
**[TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/abs/2206.14286)**
