# Go KNN
[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

This library is a simple k-NN for embeddings.
* L1Distance(Manhattan)
* L2Distance(Euclidean)
* MIPS (Maximum Inner Product Search)

---

This library is intended to be a CPU KNN search for reasonably sized datasets.

For a more advanced GPU implementation, see libraries:
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
If Tensors of different float precision are used, float64 takes priority

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
data, _ := knn.NewTensor(matrix)
fmt.Println(data.Rank)  // 2
fmt.Println(data.Shape) // [2 4]
fmt.Println(data.Type)  // float64
```

**Vectors**:
```go
vector := []float32{0.2, 0.3, 0.4, 0.5}
query, _ := k.NewTensor(vector)
fmt.Println(query.Rank)  // 1
fmt.Println(query.Shape) // [4]
fmt.Println(query.Type)  // float32
```

### Searching
**New Instance**
```go
s := &knn.New{
	Data: data,	// 2d Tensor 
	K:    2,  // Number of nearest neighbors
}
```
**Search Options**
```go
// query is a 1D Tensor
indices, values, err := s.Search(query, knn.L1) // L1 Search
indices, values, err := s.Search(query, knn.L2, 0.95) // L2 Search has an option of passing in a recall_target float64
indices, values, err := s.Search(query, knn.MIPS, 2) // MIPS has an option of passing in a bin_size int
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

	data, _ := knn.NewTensor(matrix)
	query, _ := knn.NewTensor(vector)

	s := &knn.New{
		Data: data,
		K:    2,
	}

	indices, values, err := s.Search(query, knn.MIPS, 1)
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
2024/08/10 17:17:13 [INFO] MIPS: qy=[1536], db=[8 1536], k=2, bs=1
Query Sentence: I am a scientist who enjoys fishing when I'm not in the lab.
Nearerst neighbor[0]: The scientist enjoys conducting experiments in the laboratory.
Nearerst neighbor[1]: The sailor enjoys sailing on a boat in the sea.
Indices: [5 0]
Values: [0.877427339553833 0.828193724155426]
```

## Sources:
**[TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s](https://arxiv.org/abs/2206.14286)**
