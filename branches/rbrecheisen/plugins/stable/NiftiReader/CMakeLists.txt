OPTION(PLUGINS_BUILD_NIFTIREADER "Build NIfTI Reader Plugin" ON)

if(PLUGINS_BUILD_NIFTIREADER)

set( BMIA_NIFTIREADER_PLUGIN )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    NiftiReaderPlugin.h
    NiftiReaderPlugin.cxx
    bmiaNiftiReader.h
    bmiaNiftiReader.cxx
)

link_directories(${BMIA_LIBS_BINARY_DIR})

qt4_wrap_cpp( MOC_SRCS NiftiReaderPlugin.h )

add_library( NiftiReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( NiftiReaderPlugin bmia_plugin bmia_data vtkIO vtkNIfTI bmia_libs )

endif(PLUGINS_BUILD_NIFTIREADER)