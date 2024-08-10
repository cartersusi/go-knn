package knn

import "log"

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

var logLevels = map[int]string{
	Info:    "INFO",
	Debug:   "DEBUG",
	Warning: "WARNING",
	Error:   "ERROR",
}

var logColors = map[int]string{
	Info:    green,
	Debug:   blue,
	Warning: yellow,
	Error:   red,
}

func Log(msg string, level int) {
	color, exists := logColors[level]
	if !exists {
		color = reset
	}
	log.Printf("%s[%s] %s%s\n", color, logLevels[level], msg, reset)
}
