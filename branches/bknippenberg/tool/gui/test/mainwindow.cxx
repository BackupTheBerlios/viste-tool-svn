/*
 * mainwindow.cxx
 *
 * 2009-12-22	Tim Peeters
 * - First version	
 *
 * 2010-11-29	Evert van Aart
 * - Added support for loading "profile.ini".
 *
 */

#include <QtGui/QApplication>
#include "gui/MainWindow.h"
#include "core/Core.h"
#include "plugin/Manager.h"

#include <QtGui/QFileDialog> // for QDir. XXX: Maybe other (smaller) include can work too?
//#include "DTITool.h"

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  bmia::Core* core = new bmia::Core();
  bmia::gui::MainWindow* mw = new bmia::gui::MainWindow(core);

  // load plugins
  // TODO: make this a function of the plugin manager.

  // Get the current application path
  QDir pluginsDir = QDir(qApp->applicationDirPath());

  // Try to move down to the "plugins" folder
  // Evert: Why is this here? My application folder does not contain a "plugins" subfolder,
  // so this action always fails. I guess it's relatively harmless, but if it's not needed
  // I'd rather get rid of it.
//  pluginsDir.cd("plugins");

  // Try to read the list of plugins that should be loaded from the "profile.ini" file.
//  if (!(core->plugin()->readProfile(pluginsDir)))
 // {
	  // If this fails (usually because "profile.ini" does not yet exist), read all DLLs
	  // in the current directory. At the end of this function, a default "profile.ini"
	  // file is created.

//	  core->plugin()->readAll(pluginsDir);
//  }
	  core->loadSettings();
  mw->refreshPluginList();
  mw->show();

  int appResult = app.exec();

//	delete core;
//	delete mw;

	return appResult;
}
