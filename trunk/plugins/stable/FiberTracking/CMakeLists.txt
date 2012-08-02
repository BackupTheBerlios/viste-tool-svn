# Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions
# are met:
# 
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
# 
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the 
#     distribution.
# 
#   - Neither the name of Eindhoven University of Technology nor the
#     names of its contributors may be used to endorse or promote 
#     products derived from this software without specific prior 
#     written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

OPTION(PLUGINS_BUILD_FIBERTRACKINGCPU "Build Fiber Tracking Plugin (CPU-only version)" ON)
OPTION(PLUGINS_BUILD_FIBERTRACKINGGPU "Build Fiber Tracking Plugin (CUDA version)" OFF)

if (PLUGINS_BUILD_FIBERTRACKINGCPU OR PLUGINS_BUILD_FIBERTRACKINGGPU)

set( BMIA_FIBERTRACKING_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(MAIN_SRCS
	FiberTrackingPlugin.cxx
    FiberTrackingPlugin.h
	vtkFiberTrackingFilter.cxx
	vtkFiberTrackingFilter.h	
 	vtkFiberTrackingWVSFilter.cxx
	vtkFiberTrackingWVSFilter.h	
 	FiberTrackingPlugin_WVS.cxx
	FiberTrackingPlugin_Streamlines.cxx
	streamlineTracker.cxx
	streamlineTracker.h
	CDistanceVolume.cxx
	CDistanceVolume.h
)

set(DEFAULT_SRCS ${MAIN_SRCS}
	FiberTrackingPlugin_Config_Default.cxx
	FiberTrackingPlugin_Geodesic.cxx
	geodesicFiberNeighborhood.cxx
	geodesicFiberNeighborhood.h
	geodesicPreProcessor.cxx
	geodesicPreProcessor.h
	geodesicFiberTracker.cxx
	geodesicFiberTracker.h
	vtkFiberTrackingGeodesicFilter.cxx
	vtkFiberTrackingGeodesicFilter.h
)

set(CUDA_SRCS ${MAIN_SRCS} 
	FiberTrackingPlugin_Config_CUDA.cxx
	FiberTrackingPlugin_Geodesic_CUDA.cxx
	vtkFiberTrackingGeodesicFilter_CUDA.cxx
	vtkFiberTrackingGeodesicFilter_CUDA.h
	geodesicFiberTracker_CUDA.cu
	geodesicFiberTracker_CUDA.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_ui(UiHeaders FiberTracking.ui)
qt4_wrap_cpp(MOC_SRCS FiberTrackingPlugin.h)
set_source_files_properties(FiberTrackingPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

if (PLUGINS_BUILD_FIBERTRACKINGCPU)
	add_library( FiberTrackingPlugin SHARED ${DEFAULT_SRCS} ${MOC_SRCS} )
	target_link_libraries( FiberTrackingPlugin vtkTensorMath vtkHARDI bmia_libs )
endif (PLUGINS_BUILD_FIBERTRACKINGCPU)

if (PLUGINS_BUILD_FIBERTRACKINGGPU)
	add_library( FiberTrackingPluginCUDA SHARED ${CUDA_SRCS} ${MOC_SRCS} )
	target_link_libraries( FiberTrackingPluginCUDA vtkTensorMath vtkHARDI bmia_libs )
endif (PLUGINS_BUILD_FIBERTRACKINGGPU)

endif(PLUGINS_BUILD_FIBERTRACKINGCPU OR PLUGINS_BUILD_FIBERTRACKINGGPU)