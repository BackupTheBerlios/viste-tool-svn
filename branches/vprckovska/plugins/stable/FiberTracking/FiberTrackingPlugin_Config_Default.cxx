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
#include "vtkFiberTrackingGeodesicFilter.h"


namespace bmia {


//---------------------------[ isCUDASupported ]---------------------------\\

bool FiberTrackingPlugin::isCUDASupported()
{
	return true;
}


//---------------------------[ changePluginName ]--------------------------\\

void FiberTrackingPlugin::changePluginName()
{
	// Do nothing here, the default name is okay
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libFiberTrackingPlugin, bmia::FiberTrackingPlugin)
