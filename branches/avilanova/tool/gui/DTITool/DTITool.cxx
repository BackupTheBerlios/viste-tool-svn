/*
 * DTITool.cxx
 *
 * 2009-12-22	Tim Peeters
 * - First version	
 *
 * 2010-11-29	Evert van Aart
 * - Added support for loading "profile.ini".
 *
 * 2011-08-18	Evert van Aart
 * - Renamed to "DTITool.cxx", cleaned up the code.
 *
 * 2011-10-04   Ralph Brecheisen
 * - Added specific call to 'init' for core object (see Core.h)
 */


/** Includes */

#include <QApplication>
#include <QFileDialog>
#include "gui/MainWindow.h"
#include "core/Core.h"
#include "plugin/Manager.h"


//---------------------------------[ main ]--------------------------------\\

int main(int argc, char *argv[])
{
	// Create the Qt application
	QApplication app(argc, argv);

    // Create the core object and initialize it
	bmia::Core * core = new bmia::Core();
    core->init();
  
	// Create the main window
	bmia::gui::MainWindow * mw = new bmia::gui::MainWindow(core);

	// Get the current application path
	QDir appDir = QDir(qApp->applicationDirPath());

	// Load the "settings.xml" file. This reads all profiles and serttings, and
	// automatically applies the default profile.
	core->loadSettings();

	// Update the list of plugins
	mw->refreshPluginList();

	// Show the main window
	mw->show();

	// Run the application, and get the return code
	int appResult = app.exec();

	// Destructor code should go here

	// Done, return the return code
	return appResult;
}
