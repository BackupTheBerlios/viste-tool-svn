/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * FiberTrackingPlugin.cxx
 *
 * 2011-06-23	Evert van Aart
 * - First Version. Small collection of functions and statements that configure
 *   this plugin as the "Default" version (i.e., the version without CUDA). 
 *
 */


/** Includes */

#include "FiberTrackingPlugin.h"
#include "vtkFiberTrackingGeodesicFilter_CUDA.h"
#include <cutil_inline.h>


namespace bmia {


//---------------------------[ isCUDASupported ]---------------------------\\

bool FiberTrackingPlugin::isCUDASupported()
{
	int deviceCount = 0; 

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount < 1)
		return false;

	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, 0);

	if (deviceProp.major > 999)
		return false;

	return true;
}


//---------------------------[ changePluginName ]--------------------------\\

void FiberTrackingPlugin::changePluginName()
{
	// Change name to CUDA variant
	this->name = "Fiber Tracking (CUDA-Enabled)";
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libFiberTrackingPluginCUDA, bmia::FiberTrackingPlugin)
