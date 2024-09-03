package gpu

import (
	"fmt"
)

func Check() {
	fmt.Println("Hello, World!")
	gpu, err := GpuInfo()
	if err != nil {
		fmt.Println(err)
	}
	PrintGPUInfo(gpu)
}
