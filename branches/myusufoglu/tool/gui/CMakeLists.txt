# Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions
# are met:
# 
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
# 
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the 
#     distribution.
# 
#   - Neither the name of Eindhoven University of Technology nor the
#     names of its contributors may be used to endorse or promote 
#     products derived from this software without specific prior 
#     written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/MetaCanvas)
add_subdirectory(MetaCanvas)
add_subdirectory(vISTe)

set(gui_h_moc
  MainWindow.h
  PluginDialog.h
  DataDialog.h
  SaveDialog.h
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
  SaveDialog.cxx
  ProfileManager.cxx
  DefaultProfileDialog.cxx
  SettingsDialog.cxx
  Settings_GenericPageWidget.cxx
  Settings_GeneralPageWidget.cxx
  Settings_GUIShortcutsPageWidget.cxx
)

qt4_wrap_ui(UiHeaders MainWindow.ui)
qt4_wrap_cpp(MocSrcs ${gui_h_moc})

set_source_files_properties(${gui_cxx} PROPERTIES OBJECT_DEPENDS "${UiHeaders}")

add_library(bmia_gui ${gui_cxx} ${gui_h_moc} ${gui_h_other} ${MocSrcs})
target_link_libraries(bmia_gui QVTK ${QT_LIBRARIES} bmia_core vtkMetaCanvas)

