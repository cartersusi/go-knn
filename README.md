# Go KNN GPU 


---

## For CPU Check Out
[go-knn](https://github.com/carter4299/go-knn)

[![Go Reference](https://pkg.go.dev/badge/github.com/carter4299/go-knn.svg)](https://pkg.go.dev/github.com/carter4299/go-knn)

---

## GPU Connection

```go

func GpuInfo() (GPU, error) {
	var gpu GPU

	cGpuInfo := C.run_query()
	defer C.free(unsafe.Pointer(cGpuInfo))

	if !bool(cGpuInfo.Success) {
		return gpu, fmt.Errorf("failed, could not get GPU info")
	}
	...
}
type GPU struct {
	DeviceNum                           int
	DeviceName                          string
	CudaVersion                         float32
	RuntimeVersion                      float32
	CudaCapability                      float32
	GlobalMemory                        int64
	Multiprocessors                     int
	CudaCoresPerMultiprocessor          int
	CudaCores                           int
	GpuMaxClockRate                     int
	MemoryClockRate                     int
	MemoryBusWidth                      int
	L2CacheSize                         int
	MaxTextureDimensionSize             TextureDimension
	MaxLayered1DTextureSize             Texture
	MaxLayered2DTextureSize             Texture
	TotalConstantMemory                 int64
	TotalSharedMemoryPerBlock           int64
	TotalSharedMemoryPerMultiprocessor  int64
	TotalRegistersPerBlock              int
	WarpSize                            int
	MaxThreadsPerMultiprocessor         int
	MaxThreadsPerBlock                  int
	MaxDimensionSizeOfThreadBlock       ThreadBlock
	MaxDimensionSizeOfGridSize          GridSize
	MaxMemoryPitch                      int64
	TextureAlignment                    int64
	ConcurrentCopyAndKernelExecution    bool
	RunTimeLimitOnKernels               bool
	IntegratedGpuSharingHostMemory      bool
	SupportHostPageLockedMemoryMapping  bool
	AlignmentRequirementForSurfaces     bool
	DeviceHasEccSupport                 bool
	DeviceSupportsUnifiedAddressing     bool
	DeviceSupportsManagedMemory         bool
	DeviceSupportsComputePreemption     bool
	SupportsCooperativeKernelLaunch     bool
	SupportsMultiDeviceCoopKernelLaunch bool
	DevicePciDomainIdBusIdLocationId    DevicePci
}
```

```c
GpuInfo *run_query() {
    GpuInfo *gpuInfo = new GpuInfo();

    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	...
}

typedef struct {
    bool Success;
    int DeviceNum;
    char DeviceName[256];
	int CudaDriverVersion;
    int CudaDriverVersionMinor;
    int CudaRuntimeVersion;
    int CudaRuntimeVersionMinor;
    int CudaCapabilityMajor;
    int CudaCapabilityMinor;
    char TotalGlobalMemory[256];
    int Multiprocessors;
    int CudaCoresPerMultiprocessor;
    int GpuMaxClockRate;
    int MemoryClockRate;
    int MemoryBusWidth;
    int L2CacheSize;
    int MaxTextureDimensionSize[6];
    int MaxLayered1DTextureSize[2];
    int MaxLayered2DTextureSize[3];
    int64_t TotalConstantMemory;
    int64_t TotalSharedMemoryPerBlock;
    int64_t TotalSharedMemoryPerMultiprocessor;
    int TotalRegistersPerBlock;
    int WarpSize;
    int MaxThreadsPerMultiprocessor;
    int MaxThreadsPerBlock;
    int MaxDimensionSizeOfThreadBlock[3];
    int MaxDimensionSizeOfGridSize[3];
    int64_t MaxMemoryPitch;
    int64_t TextureAlignment;
    bool ConcurrentCopyAndKernelExecution;
    bool RunTimeLimitOnKernels;
    bool IntegratedGpuSharingHostMemory;
    bool SupportHostPageLockedMemoryMapping;
    bool AlignmentRequirementForSurfaces;
    bool DeviceHasEccSupport;
    bool DeviceSupportsUnifiedAddressing;
    bool DeviceSupportsManagedMemory;
    bool DeviceSupportsComputePreemption;
    bool SupportsCooperativeKernelLaunch;
    bool SupportsMultiDeviceCoopKernelLaunch;
    int DevicePciDomainIdBusIdLocationId[3];
    } GpuInfo;
```