set(plugin_h_moc
  Plugin.h
  AdvancedPlugin.h
)

set(plugin_h_other
  DataInterfaces.h
  GUIInterfaces.h
  PluginInterfaces.h
  VisualizationInterfaces.h
  Manager.h
)

set(plugin_cxx
  Plugin.cxx
  AdvancedPlugin.cxx
  Manager.cxx
)

qt4_wrap_cpp(MOC_SRCS ${plugin_h_moc})

add_library(bmia_plugin ${plugin_cxx} ${plugin_h_moc} ${plugin_h_other} ${MOC_SRCS})
target_link_libraries(bmia_plugin bmia_core ${QT_LIBRARIES})
