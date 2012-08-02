add_executable(coredata coredata.cxx)
target_link_libraries(coredata bmia_core)
add_test(CoreData ${CMAKE_BINARY_DIR}/bin/coredata)
