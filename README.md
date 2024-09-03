# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

---

For more advanced GPU implementations, see:
* FAISS - https://github.com/facebookresearch/faiss
* Annoy - https://github.com/spotify/annoy

## Installation
```sh
go get github.com/carter4299/go-knn
```

## Usage
### Importing
```go 
import "github.com/carter4299/go-knn"
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
matrix := [][]float64{
	{0.1, 0.2, 0.3, 0.4},
	{0.4, 0.5, 0.6, 0.7},
}
m := &knn.Tensor[float64]{}
m.New(data)
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
	Data: data,	// 2d Tensor 
	Query: query // 1D Tensor
}
```

**Updating an Instance**
```go
s.Query = new_query
```

**Search Options**
* L1Distance(Manhattan)
* L2Distance(Euclidean)
* MIPS (Maximum Inner Product Search)
```go
// query is a 1D Tensor
nearest_neighbors, err := s.Search(query, knn.L1) // L1 Search
nearest_neighbors, err := s.Search(query, knn.L2) // L2 Search
nearest_neighbors, err := s.Search(query, knn.MIPS, 2) // MIPS has an option of passing in a bin_size int
```

## Example using OpenAI Ada (MIPS)
```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/carter4299/go-knn"
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
	s.ListOptions()

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
2024/08/10 17:17:13 [INFO] MIPS: qy=[1536], db=[8 1536], k=2, bs=1
Query Sentence: I am a scientist who enjoys fishing when I'm not in the lab.
Nearerst neighbor[0]: The scientist enjoys conducting experiments in the laboratory.
Nearerst neighbor[1]: The sailor enjoys sailing on a boat in the sea.
Indices: [5 0]
Values: [0.877427339553833 0.828193724155426]
```

## Sources:
**[TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/abs/2206.14286)**
