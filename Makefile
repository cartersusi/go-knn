GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

BINARY_NAME=go-knn

LDFLAGS=-ldflags="-s -w"
GCFLAGS=-gcflags="-m -l -B"

all: build

build:
	$(GOBUILD) $(LDFLAGS) $(GCFLAGS) -o $(BINARY_NAME) .

test:
	$(GOTEST) -v ./...

bench:
	$(GOTEST) -v -bench=. ./...

clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)

run: build
	./$(BINARY_NAME)

deps:
	$(GOGET) -v -t -d ./...

.PHONY: all build test benchmark clean run deps