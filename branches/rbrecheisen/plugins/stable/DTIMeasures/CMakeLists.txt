OPTION(PLUGINS_BUILD_DTIMEASURES "Build DTI Measures Plugin" ON)

if (PLUGINS_BUILD_DTIMEASURES)

set( BMIA_DTIMEASURES_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
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
target_link_libraries( DTIMeasuresPlugin vtkTensorMath bmia_libs )

endif(PLUGINS_BUILD_DTIMEASURES)