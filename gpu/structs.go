package gpu

type TextureDimension struct {
	D1 int
	D2 []int
	D3 []int
}
type Texture struct {
	D      []int
	layers int
}
type ThreadBlock struct {
	x int
	y int
	z int
}
type GridSize struct {
	x int
	y int
	z int
}

type DevicePci struct {
	DomainId   int
	BusId      int
	LocationId int
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
