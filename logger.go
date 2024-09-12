package knn

import (
	"bytes"
	"log"
	"sync"
)

const (
	reset  = "\033[0m"
	red    = "\033[31m"
	green  = "\033[32m"
	yellow = "\033[33m"
	blue   = "\033[34m"
)

const (
	Info = iota
	Debug
	Warning
	Error
)

var logLevels = [...]string{
	"INFO",
	"DEBUG",
	"WARNING",
	"ERROR",
}

var logColors = [...]string{
	green,
	blue,
	yellow,
	red,
}

var bufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

func Log(msg string, level int) {
	buf := bufferPool.Get().(*bytes.Buffer)
	defer bufferPool.Put(buf)
	buf.Reset()

	buf.WriteString(logColors[level])
	buf.WriteByte('[')
	buf.WriteString(logLevels[level])
	buf.WriteString("] ")
	buf.WriteString(msg)
	buf.WriteString(reset)
	buf.WriteByte('\n')

	log.Output(2, buf.String())
}
