PROJECT(BMIA_GPUGLYPHS_TEST)

SET(EXECUTABLES
	##  VecVol
	##  dtiglyphs
	#  dtiglyphs2
	##  shglyphs
)

INCLUDE(${BMIA_GPUGLYPHS_SOURCE_DIR}/Shaders/shaders.cmake)

FOREACH(shaderfile ${ShaderFiles})
CONFIGURE_FILE(	${BMIA_GPUGLYPHS_SOURCE_DIR}/Shaders/${shaderfile}
		${BMIA_GPUGLYPHS_TEST_BINARY_DIR}/${shaderfile}
		COPYONLY
	      )
ENDFOREACH(shaderfile)

IF(BMIA_BUILD_QT_GUI)

  SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
  SET(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
  SET(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
  SET(DESIRED_QT_VERSION ${VTK_DESIRED_QT_VERSION} CACHE FILEPATH "")

  FIND_PACKAGE(Qt)
  IF(QT_USE_FILE)
    INCLUDE(${QT_USE_FILE})
  ELSE(QT_USE_FILE)
    SET(QT_LIBRARIES ${QT_QT_LIBRARY})
  ENDIF(QT_USE_FILE)

  # Use the include path and library for Qt that is used by VTK.
  INCLUDE_DIRECTORIES( ${QT_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR}
                       ${QT_QTCORE_INCLUDE_DIR}
  		  ${CMAKE_CURRENT_BINARY_DIR}
#  		  ${CMAKE_CURRENT_SOURCE_DIR}
		   )

#  SET(UIS shglyphsgui.ui)
  QT4_WRAP_UI(UIHeaders shglyphsgui.ui)
  QT4_WRAP_CPP(MOCSrcs SHGlyphsGUI.h CHGlyphsGUI.h)

  ADD_DEFINITIONS(-DQT_GUI_LIBS -DQT_CORE_LIBS)
  SET_SOURCE_FILES_PROPERTIES(SHGlyphsGUI.cxx PROPERTIES OBJECT_DEPENDS "${UIHeaders}")
  SET_SOURCE_FILES_PROPERTIES(CHGlyphsGUI.cxx PROPERTIES OBJECT_DEPENDS "${UIHeaders}")

  #  ADD_EXECUTABLE(qshglyphs qshglyphs.cxx SHGlyphsGUI.cxx CHGlyphsGUI.cxx ${MOCSrcs})
  #TARGET_LINK_LIBRARIES(qshglyphs vtkGPUGlyphs vtkWidgets QVTK ${QT_LIBRARIES})

ENDIF(BMIA_BUILD_QT_GUI)

FOREACH(executable ${EXECUTABLES})
  ADD_EXECUTABLE(${executable} ${executable}.cxx)
  TARGET_LINK_LIBRARIES(${executable} vtkGPUGlyphs vtkWidgets)
ENDFOREACH(executable)
