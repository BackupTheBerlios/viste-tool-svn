FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})

SET (SRCS
    TestPlugin.cxx
    TestPlugin.h
    vtkInteractorStyleTrackballPositionPicker.cxx
    vtkInteractorStyleTrackballPositionPicker.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

QT4_WRAP_UI(UiHeaders TestPlugin.ui)
SET_SOURCE_FILES_PROPERTIES(TestPlugin.cxx PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

QT4_WRAP_CPP(MOC_SRCS TestPlugin.h)
ADD_LIBRARY(TestPlugin SHARED ${SRCS} ${MOC_SRCS})

TARGET_LINK_LIBRARIES(TestPlugin bmia_core bmia_plugin bmia_data)
