add_executable(attributes attributes.cxx)
target_link_libraries(attributes bmia_data)
add_test(DataAttributes ${CMAKE_BINARY_DIR}/bin/attributes)

add_executable(datamanager datamanager.cxx)
target_link_libraries(datamanager bmia_data)
add_test(DataManager ${CMAKE_BINARY_DIR}/bin/datamanager)
