set( BMIA_TRANSFERFUNCTION_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
    TransferFunctionPlugin.h
    TransferFunctionPlugin.cxx
    QAnchorItem.h
    QAnchorItem.cxx
    QCustomScene.h
    QCustomScene.cxx
    QTransferFunctionCanvas.h
    QTransferFunctionCanvas.cxx
    QLookUpTableWidget.h
    QLookUpTableWidget.cxx
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories( . ) #for the includes of the ui form
qt4_wrap_ui(UiHeaders TransferFunctionForm.ui)
qt4_wrap_cpp( MOC_SRCS TransferFunctionPlugin.h QTransferFunctionCanvas.h QCustomScene.h QLookUpTableWidget.h)

set_source_files_properties(TransferFunctionPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library( TransferFunctionPlugin SHARED ${SRCS} ${MOC_SRCS} QAnchorItem.h  )
target_link_libraries( TransferFunctionPlugin bmia_plugin vtkIO )
