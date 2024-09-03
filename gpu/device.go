package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/culib
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/culib
#cgo LDFLAGS: -L${SRCDIR}/culib
#cgo LDFLAGS: -lDeviceQuery

#include <culib.h>
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

func GpuInfo() (GPU, error) {
	var gpu GPU

	cGpuInfo := C.run_query()
	defer C.free(unsafe.Pointer(cGpuInfo))

	if !bool(cGpuInfo.Success) {
		return gpu, fmt.Errorf("failed, could not get GPU info")
	}

	gpu = GPU{
		DeviceNum:                           int(cGpuInfo.DeviceNum),
		DeviceName:                          C.GoString(&cGpuInfo.DeviceName[0]),
		CudaVersion:                         clean_version(int(cGpuInfo.CudaDriverVersion), int(cGpuInfo.CudaDriverVersionMinor)),
		RuntimeVersion:                      clean_version(int(cGpuInfo.CudaRuntimeVersion), int(cGpuInfo.CudaRuntimeVersionMinor)),
		CudaCapability:                      clean_version(int(cGpuInfo.CudaCapabilityMajor), int(cGpuInfo.CudaCapabilityMinor)),
		GlobalMemory:                        clean_mem_string(C.GoString(&cGpuInfo.TotalGlobalMemory[0])),
		Multiprocessors:                     int(cGpuInfo.Multiprocessors),
		CudaCoresPerMultiprocessor:          int(cGpuInfo.CudaCoresPerMultiprocessor),
		CudaCores:                           int(cGpuInfo.Multiprocessors) * int(cGpuInfo.CudaCoresPerMultiprocessor),
		MemoryClockRate:                     int(cGpuInfo.MemoryClockRate),
		MemoryBusWidth:                      int(cGpuInfo.MemoryBusWidth),
		L2CacheSize:                         int(cGpuInfo.L2CacheSize),
		MaxTextureDimensionSize:             clean_texture_dimension(CArraytoSlice_safe6(cGpuInfo.MaxTextureDimensionSize)),
		MaxLayered2DTextureSize:             clean_texture(CArraytoSlice_safe3(cGpuInfo.MaxLayered2DTextureSize)),
		MaxLayered1DTextureSize:             clean_texture(CArraytoSlice_safe2(cGpuInfo.MaxLayered1DTextureSize)),
		TotalConstantMemory:                 int64(cGpuInfo.TotalConstantMemory),
		TotalSharedMemoryPerBlock:           int64(cGpuInfo.TotalSharedMemoryPerBlock),
		TotalSharedMemoryPerMultiprocessor:  int64(cGpuInfo.TotalSharedMemoryPerMultiprocessor),
		TotalRegistersPerBlock:              int(cGpuInfo.TotalRegistersPerBlock),
		WarpSize:                            int(cGpuInfo.WarpSize),
		MaxThreadsPerMultiprocessor:         int(cGpuInfo.MaxThreadsPerMultiprocessor),
		MaxThreadsPerBlock:                  int(cGpuInfo.MaxThreadsPerBlock),
		MaxDimensionSizeOfThreadBlock:       clean_thread_block(CArraytoSlice_safe3(cGpuInfo.MaxDimensionSizeOfThreadBlock)),
		MaxDimensionSizeOfGridSize:          clean_grid_size(CArraytoSlice_safe3(cGpuInfo.MaxDimensionSizeOfGridSize)),
		MaxMemoryPitch:                      int64(cGpuInfo.MaxMemoryPitch),
		TextureAlignment:                    int64(cGpuInfo.TextureAlignment),
		ConcurrentCopyAndKernelExecution:    bool(cGpuInfo.ConcurrentCopyAndKernelExecution),
		RunTimeLimitOnKernels:               bool(cGpuInfo.RunTimeLimitOnKernels),
		IntegratedGpuSharingHostMemory:      bool(cGpuInfo.IntegratedGpuSharingHostMemory),
		SupportHostPageLockedMemoryMapping:  bool(cGpuInfo.SupportHostPageLockedMemoryMapping),
		AlignmentRequirementForSurfaces:     bool(cGpuInfo.AlignmentRequirementForSurfaces),
		DeviceHasEccSupport:                 bool(cGpuInfo.DeviceHasEccSupport),
		DeviceSupportsUnifiedAddressing:     bool(cGpuInfo.DeviceSupportsUnifiedAddressing),
		DeviceSupportsManagedMemory:         bool(cGpuInfo.DeviceSupportsManagedMemory),
		DeviceSupportsComputePreemption:     bool(cGpuInfo.DeviceSupportsComputePreemption),
		SupportsCooperativeKernelLaunch:     bool(cGpuInfo.SupportsCooperativeKernelLaunch),
		SupportsMultiDeviceCoopKernelLaunch: bool(cGpuInfo.SupportsMultiDeviceCoopKernelLaunch),
		DevicePciDomainIdBusIdLocationId:    clean_pci(CArraytoSlice_safe3(cGpuInfo.DevicePciDomainIdBusIdLocationId)),
	}

	return gpu, nil
}

func PrintGPUInfo(gpu GPU) {
	fmt.Printf("Device Number: %d\n", gpu.DeviceNum)
	fmt.Printf("Device Name: %s\n", gpu.DeviceName)
	fmt.Printf("CUDA Version: %.1f\n", gpu.CudaVersion)
	fmt.Printf("Runtime Version: %.1f\n", gpu.RuntimeVersion)
	fmt.Printf("CUDA Capability: %.1f\n", gpu.CudaCapability)
	fmt.Printf("Global Memory (bytes): %d\n", gpu.GlobalMemory)
	fmt.Printf("Multiprocessors: %d\n", gpu.Multiprocessors)
	fmt.Printf("CUDA Cores Per Multiprocessor: %d\n", gpu.CudaCoresPerMultiprocessor)
	fmt.Printf("Total CUDA Cores: %d\n", gpu.CudaCores)
	fmt.Printf("Memory Clock Rate (MHz): %d\n", gpu.MemoryClockRate/1000)
	fmt.Printf("Memory Bus Width (bits): %d\n", gpu.MemoryBusWidth)
	fmt.Printf("L2 Cache Size: %d\n", gpu.L2CacheSize)
	fmt.Printf("Max Texture Dimension Size: D1=%d, D2=%v, D3=%v\n", gpu.MaxTextureDimensionSize.D1, gpu.MaxTextureDimensionSize.D2, gpu.MaxTextureDimensionSize.D3)
	fmt.Printf("Max Layered 1D Texture Size: %v, Layers=%d\n", gpu.MaxLayered1DTextureSize.D, gpu.MaxLayered1DTextureSize.layers)
	fmt.Printf("Max Layered 2D Texture Size: %v, Layers=%d\n", gpu.MaxLayered2DTextureSize.D, gpu.MaxLayered2DTextureSize.layers)
	fmt.Printf("Total Constant Memory (bytes): %d\n", gpu.TotalConstantMemory)
	fmt.Printf("Total Shared Memory Per Block (bytes): %d\n", gpu.TotalSharedMemoryPerBlock)
	fmt.Printf("Total Shared Memory Per Multiprocessor (bytes): %d\n", gpu.TotalSharedMemoryPerMultiprocessor)
	fmt.Printf("Total Registers Per Block (bytes): %d\n", gpu.TotalRegistersPerBlock)
	fmt.Printf("Warp Size: %d\n", gpu.WarpSize)
	fmt.Printf("Max Threads Per Multiprocessor: %d\n", gpu.MaxThreadsPerMultiprocessor)
	fmt.Printf("Max Threads Per Block: %d\n", gpu.MaxThreadsPerBlock)
	fmt.Printf("Max Dimension Size Of Thread Block: X=%d, Y=%d, Z=%d\n", gpu.MaxDimensionSizeOfThreadBlock.x, gpu.MaxDimensionSizeOfThreadBlock.y, gpu.MaxDimensionSizeOfThreadBlock.z)
	fmt.Printf("Max Dimension Size Of Grid Size: X=%d, Y=%d, Z=%d\n", gpu.MaxDimensionSizeOfGridSize.x, gpu.MaxDimensionSizeOfGridSize.y, gpu.MaxDimensionSizeOfGridSize.z)
	fmt.Printf("Max Memory Pitch (bytes): %d\n", gpu.MaxMemoryPitch)
	fmt.Printf("Texture Alignment (bytes): %d\n", gpu.TextureAlignment)
	fmt.Printf("Concurrent Copy And Kernel Execution: %t\n", gpu.ConcurrentCopyAndKernelExecution)
	fmt.Printf("Run Time Limit On Kernels: %t\n", gpu.RunTimeLimitOnKernels)
	fmt.Printf("Integrated GPU Sharing Host Memory: %t\n", gpu.IntegratedGpuSharingHostMemory)
	fmt.Printf("Support Host Page Locked Memory Mapping: %t\n", gpu.SupportHostPageLockedMemoryMapping)
	fmt.Printf("Alignment Requirement For Surfaces: %t\n", gpu.AlignmentRequirementForSurfaces)
	fmt.Printf("Device Has Ecc Support: %t\n", gpu.DeviceHasEccSupport)
	fmt.Printf("Device Supports Unified Addressing: %t\n", gpu.DeviceSupportsUnifiedAddressing)
	fmt.Printf("Device Supports Managed Memory: %t\n", gpu.DeviceSupportsManagedMemory)
	fmt.Printf("Device Supports Compute Preemption: %t\n", gpu.DeviceSupportsComputePreemption)
	fmt.Printf("Supports Cooperative Kernel Launch: %t\n", gpu.SupportsCooperativeKernelLaunch)
	fmt.Printf("Supports MultiDevice Co-op Kernel Launch: %t\n", gpu.SupportsMultiDeviceCoopKernelLaunch)
	fmt.Printf("Device PCI Domain ID / Bus ID / location ID: DomainId=%d, BusId=%d, LocationId=%d\n", gpu.DevicePciDomainIdBusIdLocationId.DomainId, gpu.DevicePciDomainIdBusIdLocationId.BusId, gpu.DevicePciDomainIdBusIdLocationId.LocationId)
}

func clean_version(major int, minor int) float32 {
	return float32(major) + float32(minor)/10
}

func clean_mem_string(s string) int64 {
	/*   Total amount of global memory:                 7958 MBytes (8344829952 bytes)
	 */
	var ret int64
	fmt.Sscanf(strings.Split(s, "(")[1], "%d", &ret)
	return ret
}
func clean_texture_dimension(arr []int) TextureDimension {
	/* [131072 131072 65536 16384 16384 16384]
	 */
	var ret TextureDimension
	ret.D1 = arr[0]
	ret.D2 = arr[1:3]
	ret.D3 = arr[3:]
	return ret
}

func clean_texture(arr []int) Texture {
	/* 2D=[32768 32768 2048] 1D=[32768 2048]
	 */
	var ret Texture
	ret.D = arr[:len(arr)-1]
	ret.layers = arr[len(arr)-1]
	return ret

}

func clean_thread_block(arr []int) ThreadBlock {
	/* [1024 1024 64]
	 */
	var ret ThreadBlock
	ret.x = arr[0]
	ret.y = arr[1]
	ret.z = arr[2]
	return ret
}

func clean_grid_size(arr []int) GridSize {
	/* [2147483647 65535 65535]
	 */
	var ret GridSize
	ret.x = arr[0]
	ret.y = arr[1]
	ret.z = arr[2]
	return ret
}

func clean_pci(arr []int) DevicePci {
	/* [0 1 0]
	 */
	var ret DevicePci
	ret.DomainId = arr[0]
	ret.BusId = arr[1]
	ret.LocationId = arr[2]
	return ret
}

func CArraytoSlice_safe6(c_arr [6]C.int) []int {
	var ret []int
	for _, val := range c_arr {
		ret = append(ret, int(val))
	}
	return ret
}
func CArraytoSlice_safe3(c_arr [3]C.int) []int {
	var ret []int
	for _, val := range c_arr {
		ret = append(ret, int(val))
	}
	return ret
}
func CArraytoSlice_safe2(c_arr [2]C.int) []int {
	var ret []int
	for _, val := range c_arr {
		ret = append(ret, int(val))
	}
	return ret
}

/*
func CArraytoSlice_safe5(c_arr [5]C.int) []int {
	var ret []int
	for _, val := range c_arr {
		ret = append(ret, int(val))
	}
	return ret
}
func CArraytoSlice_safe4(c_arr [4]C.int) []int {
	var ret []int
	for _, val := range c_arr {
		ret = append(ret, int(val))
	}
	return ret
}

*/
