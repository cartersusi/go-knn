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
#ifndef _CULIB_H_
#define _CULIB_H_

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

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

GpuInfo *run_query();

#endif