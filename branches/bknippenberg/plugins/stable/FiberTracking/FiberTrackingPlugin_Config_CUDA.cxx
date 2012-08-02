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
