OPTION(PLUGINS_BUILD_HARDICONVERTER "Build HARDI Converter Plugin" ON)

if(PLUGINS_BUILD_HARDICONVERTER)

set(BMIA_HARDICONVERTER_PLUGIN)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set (SRCS
HARDIConverterPlugin.cxx
HARDIConverterPlugin.h
vtkSH2DSFFilter.cxx
vtkSH2DSFFilter.h
)

include_directories(${BMIA_LIBS_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

qt4_wrap_ui(UiHeaders HARDIConverter.ui)
qt4_wrap_cpp(MOC_SRCS HARDIConverterPlugin.h)
set_source_files_properties(HARDIConverterPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

link_directories(${BMIA_LIBS_BINARY_DIR})

add_library(HARDIConverterPlugin SHARED ${SRCS} ${MOC_SRCS})
target_link_libraries(HARDIConverterPlugin bmia_core bmia_plugin bmia_data vtkHARDI)

endif(PLUGINS_BUILD_HARDICONVERTER)