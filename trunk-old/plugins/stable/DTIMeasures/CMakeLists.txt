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

OPTION(PLUGINS_BUILD_DTIMEASURES "Build DTI Measures Plugin" ON)

if (PLUGINS_BUILD_DTIMEASURES)

set( BMIA_DTIMEASURES_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	vtkTensorToScalarFilter.h
	vtkTensorToScalarFilter.cxx
	vtkTensorToInvariantFilter.h
	vtkTensorToInvariantFilter.cxx
    vtkTensorToEigensystemFilter.h
    vtkTensorToEigensystemFilter.cxx
    vtkEigenvaluesToScalarFilter.h
    vtkEigenvaluesToScalarFilter.cxx
    vtkEigenvaluesToAnisotropyFilter.h
    vtkEigenvaluesToAnisotropyFilter.cxx
    DTIMeasuresPlugin.h
    DTIMeasuresPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS DTIMeasuresPlugin.h )

add_library( DTIMeasuresPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( DTIMeasuresPlugin vtkTensorMath bmia_libs vtkVolumeRendering vtkIO vtkWidgets vtkGraphics vtkRendering)

endif(PLUGINS_BUILD_DTIMEASURES)