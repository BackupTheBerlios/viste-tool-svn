CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( DicomConverterPlugin )

SET( BMIA_DICOMCONVERTER_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	/usr/local/include/gsl
	/usr/local/include/gdcm-2.0
	)

LINK_DIRECTORIES(
	/usr/lib
	/usr/local/lib
	)

SET( SRCS
	DicomConverterPlugin.cxx
	DicomConverterPlugin.h
	DTIBootstrapConverter.cxx
	DTIBootstrapConverter.h
	DTIConfig.cxx
	DTIConfig.h
	DTIData2TensorConverter2.cxx
	DTIData2TensorConverter2.h
	DTIDICOMPhilipsReader.cxx
	DTIDICOMPhilipsReader.h
	DTIDICOMReader.cxx
	DTIDICOMReader.h
	DTIDICOMReader.cxx
	DTIDICOMReader.h
	DTIDICOMSlice.cxx
	DTIDICOMSlice.h
	DTIDICOMSliceGroup.cxx
	DTIDICOMSliceGroup.h
	DTIGradientReader.cxx
	DTIGradientReader.h
	DTISlice.cxx
	DTISlice.h
	DTISliceGroup.cxx
	DTISliceGroup.h
	DTITensor2DtiToolConverter.cxx
	DTITensor2DtiToolConverter.h
	DTITensorSlice.cxx
	DTITensorSlice.h
	DTIUtils.cxx
	DTIUtils.h
	DTIVectorSlice.cxx
	DTIVectorSlice.h
	)

QT4_WRAP_UI( UIC_SRCS
	DicomConverterPlugin.ui
	)

QT4_WRAP_CPP( MOC_SRCS
	DicomConverterPlugin.h
	)

ADD_LIBRARY( DicomConverterPlugin SHARED
	${SRCS} ${UIC_SRCS} ${MOC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( DicomConverterPlugin
	gdcmCommon gdcmDICT gdcmcharls gdcmIOD gdcmDSED gdcmjpeg12
	gdcmjpeg16 gdcmjpeg8 gdcmMSFF gdcmopenjpeg gdcmuuid
	gsl gslcblas
	bmia_plugin
	)
