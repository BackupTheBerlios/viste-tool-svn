CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
SET( PROJECT_NAME DicomReaderPlugin )
SET( BMIA_DICOMREADER_PLUGIN )
PROJECT( ${PROJECT_NAME} )

#----------------------------------------------------------
# Find QT package
#----------------------------------------------------------
FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

#----------------------------------------------------------
# Find GDCM package
#----------------------------------------------------------
FIND_PACKAGE( GDCM REQUIRED )
INCLUDE( ${GDCM_USE_FILE} )

#----------------------------------------------------------
# Include and link directories
#----------------------------------------------------------
INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	)

#----------------------------------------------------------
# Run MOC compiler
#----------------------------------------------------------
QT4_WRAP_CPP( SOURCES_MOC
	DicomReaderPlugin.h
	)

#----------------------------------------------------------
# Add sources and headers to library
#----------------------------------------------------------
ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SOURCES_MOC}
	DicomReaderPlugin.cxx
	DicomReaderPlugin.h
	)
	
#----------------------------------------------------------
# Specify link libraries
#----------------------------------------------------------
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
	gdcmCommon gdcmDICT gdcmIOD gdcmDSED gdcmjpeg12
	gdcmjpeg16 gdcmjpeg8 gdcmMSFF gdcmopenjpeg gdcmzlib gdcmexpat 
	vtkgdcm vtkIO
	bmia_plugin 
	)
