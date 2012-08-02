include_directories(${CMAKE_CURRENT_SOURCE_DIR}/MetaCanvas)
add_subdirectory(MetaCanvas)

set(gui_h_moc
  MainWindow.h
  PluginDialog.h
  DataDialog.h
  ProfileManager.h
  DefaultProfileDialog.h
  SettingsDialog.h
  Settings_GenericPageWidget.h
  Settings_GeneralPageWidget.h
  Settings_GUIShortcutsPageWidget.h
)

set(gui_h_other
  MainWindowInterface.h
)

set(gui_cxx
  MainWindow.cxx
  PluginDialog.cxx
  DataDialog.cxx
  ProfileManager.cxx
  DefaultProfileDialog.cxx
  SettingsDialog.cxx
  Settings_GenericPageWidget.cxx
  Settings_GeneralPageWidget.cxx
  Settings_GUIShortcutsPageWidget.cxx
)

qt4_wrap_ui(UiHeaders mainwindow.ui)
qt4_wrap_cpp(MocSrcs ${gui_h_moc})

set_source_files_properties(${gui_cxx} PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library(bmia_gui ${gui_cxx} ${gui_h_moc} ${gui_h_other} ${MocSrcs})
target_link_libraries(bmia_gui QVTK ${QT_LIBRARIES} bmia_core vtkMetaCanvas)

if (BMIA_BUILD_TESTS)
  add_subdirectory(test)
endif (BMIA_BUILD_TESTS)
