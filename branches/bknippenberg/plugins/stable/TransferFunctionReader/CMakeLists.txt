set( BMIA_TRANSFERFUNCTIONREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    TransferFunctionReaderPlugin.h
    TransferFunctionReaderPlugin.cxx
)

qt4_wrap_cpp( MOC_SRCS TransferFunctionReaderPlugin.h )

add_library( TransferFunctionReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( TransferFunctionReaderPlugin bmia_plugin bmia_data vtkIO )
