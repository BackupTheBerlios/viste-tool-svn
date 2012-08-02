set(data_files
Attributes.h
Attributes.cxx
DataSet.h
DataSet.cxx
Consumer.h
Reader.h
Manager.h
Manager.cxx
)

add_library(bmia_data ${data_files})
target_link_libraries(bmia_data vtkFiltering ${QT_LIBRARIES})
