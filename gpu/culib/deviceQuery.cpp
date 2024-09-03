/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>
#include <memory>
#include <string>
#include <stdint.h>
#include <stdbool.h>

int *pArgc = NULL;
char **pArgv = NULL;


int64_t go_convert_size_t(size_t size) {
    return static_cast<int64_t>(size);
}
bool go_convert_bool(int value) {
    return value != 0;
}

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    //exit(EXIT_FAILURE);
    return;
  }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
extern "C" {

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

    GpuInfo *run_query() {
        GpuInfo *gpuInfo = new GpuInfo();

        printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceCount returned %d\n-> %s\n",static_cast<int>(error_id), cudaGetErrorString(error_id));
            printf("Result = FAIL\n");        

            // removed from original code and replaced with return, dont know the full impact of this
            //exit(EXIT_FAILURE);
            gpuInfo->Success = false;
            return gpuInfo;

        }

        // This function call returns 0 if there are no CUDA capable devices.
        if (deviceCount == 0) {
            printf("There are no available device(s) that support CUDA\n");

            gpuInfo->Success = false;
            return gpuInfo;
        } else {
            printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        }

        int dev, driverVersion = 0, runtimeVersion = 0;

        for (dev = 0; dev < deviceCount; ++dev) {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);

            printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
            gpuInfo->DeviceNum = dev;
            strcpy(gpuInfo->DeviceName, deviceProp.name);


            // Console log
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
                driverVersion / 1000, (driverVersion % 100) / 10,
                runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            gpuInfo->CudaDriverVersion = driverVersion / 1000;
            gpuInfo->CudaDriverVersionMinor = (driverVersion % 100) / 10;
            gpuInfo->CudaRuntimeVersion = runtimeVersion / 1000;
            gpuInfo->CudaRuntimeVersionMinor = (runtimeVersion % 100) / 10;

            printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                deviceProp.major, deviceProp.minor);
            gpuInfo->CudaCapabilityMajor = deviceProp.major;
            gpuInfo->CudaCapabilityMinor = deviceProp.minor;


            char msg[256];
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            sprintf_s(msg, sizeof(msg),
                    "  Total amount of global memory:                 %.0f MBytes "
                    "(%llu bytes)\n",
                    static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                    (unsigned long long)deviceProp.totalGlobalMem);
        #else
            snprintf(msg, sizeof(msg),
                    "  Total amount of global memory:                 %.0f MBytes "
                    "(%llu bytes)\n",
                    static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                    (unsigned long long)deviceProp.totalGlobalMem);
        #endif
            printf("%s", msg);
            strcpy(gpuInfo->TotalGlobalMemory, msg);

            printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
                deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
                    deviceProp.multiProcessorCount);
            gpuInfo->Multiprocessors = deviceProp.multiProcessorCount;
            gpuInfo->CudaCoresPerMultiprocessor = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

            printf(
                "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
                "GHz)\n",
                deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
            gpuInfo->GpuMaxClockRate = deviceProp.clockRate;

        #if CUDART_VERSION >= 5000
            // This is supported in CUDA 5.0 (runtime API device properties)
            printf("  Memory Clock rate:                             %.0f Mhz\n",
                deviceProp.memoryClockRate * 1e-3f);
            gpuInfo->MemoryClockRate = deviceProp.memoryClockRate;
            printf("  Memory Bus Width:                              %d-bit\n",
                deviceProp.memoryBusWidth);
            gpuInfo->MemoryBusWidth = deviceProp.memoryBusWidth;
            

            if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                    deviceProp.l2CacheSize);
            }
            gpuInfo->L2CacheSize = deviceProp.l2CacheSize;

        #else
            // This only available in CUDA 4.0-4.2 (but these were only exposed in the
            // CUDA Driver API)
            int memoryClock;
            getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                dev);
            printf("  Memory Clock rate:                             %.0f Mhz\n",
                memoryClock * 1e-3f);
            int memBusWidth;
            getCudaAttribute<int>(&memBusWidth,
                                CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
            printf("  Memory Bus Width:                              %d-bit\n",
                memBusWidth);
            int L2CacheSize;
            getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

            if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                    L2CacheSize);
            }

        #endif

            printf(
                "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
                "%d), 3D=(%d, %d, %d)\n",
                deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
                deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
                deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
            gpuInfo->MaxTextureDimensionSize[0] = deviceProp.maxTexture1D;
            gpuInfo->MaxTextureDimensionSize[1] = deviceProp.maxTexture2D[0];
            gpuInfo->MaxTextureDimensionSize[2] = deviceProp.maxTexture2D[1];
            gpuInfo->MaxTextureDimensionSize[3] = deviceProp.maxTexture3D[0];
            gpuInfo->MaxTextureDimensionSize[4] = deviceProp.maxTexture3D[1];
            gpuInfo->MaxTextureDimensionSize[5] = deviceProp.maxTexture3D[2];

            printf(
                "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
                deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
            gpuInfo->MaxLayered1DTextureSize[0] = deviceProp.maxTexture1DLayered[0];
            gpuInfo->MaxLayered1DTextureSize[1] = deviceProp.maxTexture1DLayered[1];

            printf(
                "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
                "layers\n",
                deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
                deviceProp.maxTexture2DLayered[2]);
            gpuInfo->MaxLayered2DTextureSize[0] = deviceProp.maxTexture2DLayered[0];
            gpuInfo->MaxLayered2DTextureSize[1] = deviceProp.maxTexture2DLayered[1];
            gpuInfo->MaxLayered2DTextureSize[2] = deviceProp.maxTexture2DLayered[2];

            printf("  Total amount of constant memory:               %zu bytes\n",
                deviceProp.totalConstMem);
            gpuInfo->TotalConstantMemory = go_convert_size_t(deviceProp.totalConstMem);
            
            printf("  Total amount of shared memory per block:       %zu bytes\n",
                deviceProp.sharedMemPerBlock);
            gpuInfo->TotalSharedMemoryPerBlock = go_convert_size_t(deviceProp.sharedMemPerBlock);
            
            printf("  Total shared memory per multiprocessor:        %zu bytes\n",
                deviceProp.sharedMemPerMultiprocessor);
            gpuInfo->TotalSharedMemoryPerMultiprocessor = go_convert_size_t(deviceProp.sharedMemPerMultiprocessor);
            
            printf("  Total number of registers available per block: %d\n",
                deviceProp.regsPerBlock);
            gpuInfo->TotalRegistersPerBlock = deviceProp.regsPerBlock;
            
            printf("  Warp size:                                     %d\n",
                deviceProp.warpSize);
            gpuInfo->WarpSize = deviceProp.warpSize;
            
            printf("  Maximum number of threads per multiprocessor:  %d\n",
                deviceProp.maxThreadsPerMultiProcessor);
            gpuInfo->MaxThreadsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor;
            
            printf("  Maximum number of threads per block:           %d\n",
                deviceProp.maxThreadsPerBlock);
            gpuInfo->MaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
            
            printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
            gpuInfo->MaxDimensionSizeOfThreadBlock[0] = deviceProp.maxThreadsDim[0];
            gpuInfo->MaxDimensionSizeOfThreadBlock[1] = deviceProp.maxThreadsDim[1];
            gpuInfo->MaxDimensionSizeOfThreadBlock[2] = deviceProp.maxThreadsDim[2];
            
            printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
            gpuInfo->MaxDimensionSizeOfGridSize[0] = deviceProp.maxGridSize[0];
            gpuInfo->MaxDimensionSizeOfGridSize[1] = deviceProp.maxGridSize[1];
            gpuInfo->MaxDimensionSizeOfGridSize[2] = deviceProp.maxGridSize[2];
            
            printf("  Maximum memory pitch:                          %zu bytes\n",
                deviceProp.memPitch);
            gpuInfo->MaxMemoryPitch = go_convert_size_t(deviceProp.memPitch);
            
            printf("  Texture alignment:                             %zu bytes\n",
                deviceProp.textureAlignment);
            gpuInfo->TextureAlignment = go_convert_size_t(deviceProp.textureAlignment);
            
            printf(
                "  Concurrent copy and kernel execution:          %s with %d copy "
                "engine(s)\n",
                (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
            gpuInfo->ConcurrentCopyAndKernelExecution = go_convert_bool(deviceProp.deviceOverlap);
            
            printf("  Run time limit on kernels:                     %s\n",
                deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
            gpuInfo->RunTimeLimitOnKernels = go_convert_bool(deviceProp.kernelExecTimeoutEnabled);
            
            printf("  Integrated GPU sharing Host Memory:            %s\n",
                deviceProp.integrated ? "Yes" : "No");
            gpuInfo->IntegratedGpuSharingHostMemory = go_convert_bool(deviceProp.integrated);
            
            printf("  Support host page-locked memory mapping:       %s\n",
                deviceProp.canMapHostMemory ? "Yes" : "No");
            gpuInfo->SupportHostPageLockedMemoryMapping = go_convert_bool(deviceProp.canMapHostMemory);

            printf("  Alignment requirement for Surfaces:            %s\n",
                deviceProp.surfaceAlignment ? "Yes" : "No");
            gpuInfo->AlignmentRequirementForSurfaces = go_convert_bool(deviceProp.surfaceAlignment);

            printf("  Device has ECC support:                        %s\n",
                deviceProp.ECCEnabled ? "Enabled" : "Disabled");
            gpuInfo->DeviceHasEccSupport = go_convert_bool(deviceProp.ECCEnabled);
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
                deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                        : "WDDM (Windows Display Driver Model)");
        #endif
            printf("  Device supports Unified Addressing (UVA):      %s\n",
                deviceProp.unifiedAddressing ? "Yes" : "No");
            gpuInfo->DeviceSupportsUnifiedAddressing = go_convert_bool(deviceProp.unifiedAddressing);

            printf("  Device supports Managed Memory:                %s\n",
            
                deviceProp.managedMemory ? "Yes" : "No");
            gpuInfo->DeviceSupportsManagedMemory = go_convert_bool(deviceProp.managedMemory);
            
            printf("  Device supports Compute Preemption:            %s\n",
                deviceProp.computePreemptionSupported ? "Yes" : "No");
            gpuInfo->DeviceSupportsComputePreemption = go_convert_bool(deviceProp.computePreemptionSupported);
            
            printf("  Supports Cooperative Kernel Launch:            %s\n",
                deviceProp.cooperativeLaunch ? "Yes" : "No");
            gpuInfo->SupportsCooperativeKernelLaunch = go_convert_bool(deviceProp.cooperativeLaunch);
            
            printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
                deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
            gpuInfo->SupportsMultiDeviceCoopKernelLaunch = go_convert_bool(deviceProp.cooperativeMultiDeviceLaunch);
            
            printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
                deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
            gpuInfo->DevicePciDomainIdBusIdLocationId[0] = deviceProp.pciDomainID;
            gpuInfo->DevicePciDomainIdBusIdLocationId[1] = deviceProp.pciBusID;
            gpuInfo->DevicePciDomainIdBusIdLocationId[2] = deviceProp.pciDeviceID;


            const char *sComputeMode[] = {
                "Default (multiple host threads can use ::cudaSetDevice() with device "
                "simultaneously)",
                "Exclusive (only one host thread in one process is able to use "
                "::cudaSetDevice() with this device)",
                "Prohibited (no host thread can use ::cudaSetDevice() with this "
                "device)",
                "Exclusive Process (many threads in one process is able to use "
                "::cudaSetDevice() with this device)",
                "Unknown", NULL};
            printf("  Compute Mode:\n");
            printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
        }

        // If there are 2 or more GPUs, query to determine whether RDMA is supported
        if (deviceCount >= 2) {
            cudaDeviceProp prop[64];
            int gpuid[64];  // we want to find the first two GPUs that can support P2P
            int gpu_p2p_count = 0;

            for (int i = 0; i < deviceCount; i++) {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                && prop[i].tccDriver
        #endif
                ) {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
            }

            // Show all the combinations of support P2P GPUs
            int can_access_peer;

            if (gpu_p2p_count >= 2) {
            for (int i = 0; i < gpu_p2p_count; i++) {
                for (int j = 0; j < gpu_p2p_count; j++) {
                if (gpuid[i] == gpuid[j]) {
                    continue;
                }
                checkCudaErrors(
                    cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                        prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                        can_access_peer ? "Yes" : "No");
                }
            }
            }
        }

        // csv masterlog info
        // *****************************
        // exe and CUDA driver name
        printf("\n");
        std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
        char cTemp[16];

        // driver version
        sProfileString += ", CUDA Driver Version = ";
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
                    (driverVersion % 100) / 10);
        #else
        snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
                (driverVersion % 100) / 10);
        #endif
        sProfileString += cTemp;

        // Runtime version
        sProfileString += ", CUDA Runtime Version = ";
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000,
                    (runtimeVersion % 100) / 10);
        #else
        snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
                (runtimeVersion % 100) / 10);
        #endif
        sProfileString += cTemp;

        // Device count
        sProfileString += ", NumDevs = ";
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(cTemp, 10, "%d", deviceCount);
        #else
        snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
        #endif
        sProfileString += cTemp;
        sProfileString += "\n";
        printf("%s", sProfileString.c_str());

        printf("Result = PASS\n");

        // removed from original code and replaced with return, dont know the full impact of this
        //exit(EXIT_SUCCESS);
        
        gpuInfo->Success = true;
        return gpuInfo;
    }
}