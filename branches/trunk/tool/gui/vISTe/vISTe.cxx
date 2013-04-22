/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
