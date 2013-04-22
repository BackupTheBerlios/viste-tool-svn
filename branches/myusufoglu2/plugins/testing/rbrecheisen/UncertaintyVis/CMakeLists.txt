cmake_minimum_required( VERSION 2.6 )
project( UncertaintyVisPlugin )

set( BMIA_UNCERTAINTYVIS_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../../../tool/gui/MetaCanvas
	)

set(SRCS
	UncertaintyVisPlugin.cxx
	UncertaintyVisPlugin.h
    QConfidenceHistogramWidget.cxx
    QConfidenceHistogramWidget.h
    vtkConfidenceHistogram.cxx
    vtkConfidenceHistogram.h
	vtkConfidenceInterval.cxx
	vtkConfidenceInterval.h
	vtkConfidenceIntervalProperties.cxx
	vtkConfidenceIntervalProperties.h
	vtkConfidenceTable.cxx
	vtkConfidenceTable.h
	vtkDTIComponentReader.cxx
	vtkDTIComponentReader.h
	vtkFiberConfidenceMapper.cxx
	vtkFiberConfidenceMapper.h
	vtkFiberConfidenceMapperShaders.h
	vtkROIWidget.cxx
	vtkROIWidget.h
	)

qt4_wrap_cpp( MOC_SRCS
	UncertaintyVisPlugin.h
	QConfidenceHistogramWidget.h
	)

add_library( UncertaintyVisPlugin SHARED 
    ${SRCS} ${MOC_SRCS} 
    )
    
target_link_libraries( UncertaintyVisPlugin
	bmia_plugin
	)