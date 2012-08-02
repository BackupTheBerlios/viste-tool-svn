set(core_h
  vtkTextProgressCommand.h
  UserOutput.h
  SimpleCoreInterface.h
  Core.h
  QVTKProgressCommand.h
  XMLSettingsReader.h
  XMLSettingsWriter.h
  DTIToolProfile.h
  DTIToolSettings.h
)

set(core_cxx
  vtkTextProgressCommand.cxx
  UserOutput.cxx
  Core.cxx
  QVTKProgressCommand.cxx
  XMLSettingsReader.cxx
  XMLSettingsWriter.cxx
  DTIToolProfile.cxx
  DTIToolSettings.cxx
)

add_library(bmia_core ${core_cxx} ${core_h})
target_link_libraries(bmia_core bmia_data bmia_plugin bmia_gui)
