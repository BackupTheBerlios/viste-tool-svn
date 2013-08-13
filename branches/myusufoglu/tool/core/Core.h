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
 * Core.h 
 *
 * 2009-11-05	Tim Peeters
 * - First version
 *
 * 2010-03-17	Tim Peeters
 * - No more singleton.
 *
 * 2011-03-18	Evert van Aart
 * - Added support for reading and writing "settings.xml" using XML readers/writers.
 *
 * 2011-04-06	Evert van Aart
 * - Added "getDataDirectory".
 *
 * 2011-04-14	Evert van Aart
 * - Added "disableRendering" and "enableRendering".
 *
 * 2011-10-05   Ralph Brecheisen
 * - Added 'init' for initializing the core object. Previous version of core class
 * was passing 'this' reference to plugin manager without the constructor having
 * finished. This leads to unpredictable effects when the plugin manager checks
 * the NULL-ness of the core.
 */


#ifndef bmia_Core_h
#define bmia_Core_h


/** Includes - Qt */

#include <QMessageBox>
#include <QtDebug>
#include <QApplication>

/** Includes - VTK */

#include <vtkRenderWindow.h>

/** Includes - Custom Files */

#include "SimpleCoreInterface.h"
#include "XMLSettingsReader.h"
#include "XMLSettingsWriter.h"
#include "DTIToolProfile.h"
#include "DTIToolSettings.h"
#include "UserOutput.h"
#include "gui/MetaCanvas/vtkMedicalCanvas.h"


namespace bmia {


namespace plugin 
{
	class Manager;
}

namespace data
{
	class Manager;
}

namespace gui
{
	class MainWindowInterface;
}


/** Core of the DTITool. This class itself does not do much, except keeping track
	of the different components of the core classes (like the plugin manager and
	the main window). It also takes care of loading the settings file, and applying
	the settings stored in it. Most classes can only access this core through its
	interface class, which heavily restricts access.
*/

class Core : public SimpleCoreInterface
{
	public:
    
		/** Constructor. */

		Core();

		/** Destructor. */
    
		~Core();

        /** Initialize the core */

        void init();

		/** Return the data manager. */

		data::Manager * data();

		/** Return the plugin manager. */

        plugin::Manager * plugin();
    
		/** Return the user output class. */

		UserOutput * out();

		/** Return the interface of the main window. */

		gui::MainWindowInterface * gui();

		/** Return the canvas of the main window. */

		vtkMedicalCanvas * canvas();

		/** Set a pointer to (the interface of) the main window. Called by the 
			constructor of the main window class. 
			@param mwi		Interface of the main window. */

		void setMainWindow(gui::MainWindowInterface * mwi);

		/** Set the render window pointer. Called by the constructor of the main
			window class. 
			@param rw		Render window of the main window. */

		void setRenderWindow(vtkRenderWindow * rw);

		/** Set the pointer to the canvas of the main window, on which the 
			visualization components are drawn. Called by the constructor of the
			main window class. 
			@param canvas	Medical canvas. */

		void setMedicalCanvas(vtkMedicalCanvas * canvas);

		/** Load the settings and profiles from the "settings.xml" file, and apply them. */

		void loadSettings();

		/** Redraw the scene. */
    
		void render();


		/** List of all profiles (plugins and files loaded at start-up). */

		QList<DTIToolProfile *> profiles;

		/** Return the default data directory. Can be used by reader plugins to
			set the initial directory of file dialogs. By default, uses the data
			directory of the active profile; if no active profile has been set,
			the application directory is used instead. */

		QDir getDataDirectory();

		/** Turn off rendering. If rendering is turned off, any call to the "render"
			function will immediately return. Extreme care should be taken when using
			this function: Every call to "disableRendering" should be followed by a
			call to "enableRendering" eventually! The purpose of this function is to
			avoid redundant re-renders when changing a large amount of data. For example,
			switching to a new image in the Planes plugin updates the three slice actors
			and the three plane seed point sets; each of these updates can trigger a
			render call in one (or more!) other plugins. Therefore, we turn off rendering
			before we switch to the new image, and re-enable when we're done (followed
			by a single render call. */

		void disableRendering()
		{
			allowRendering = false;
		}

		/** Turn on rendering. See notes for "disableRendering". */

		void enableRendering()
		{
			allowRendering = true;
		}

		/** Apply the DTITool settings. After applying the settings, this function
			will re-write the "settings.xml" file, using the new settings. */

		void applySettings();

		/** General DTITool settings. */
	
		DTIToolSettings * settings;

	private:

		data::Manager * dataManager;			/**< Data manager. */
		plugin::Manager * pluginManager;		/**< Plugin manager. */
		UserOutput * userOutput;				/**< User output class. */
		gui::MainWindowInterface * mainWindow;	/**< Interface of the main window. */
		vtkRenderWindow * renderWindow;			/**< Render window. */
		vtkMedicalCanvas * medicalCanvas;		/**< Medical canvas of the main window. */

		/** Pointer to the default profile (i.e., the profile that was or will be
			loaded at start-up). */
	
		DTIToolProfile * defaultProfile;

		/** Specifies whether or not calls to the "render" function will be accepted.
			See notes for "disableRendering" for details. */

		bool allowRendering;

}; // class Core


} // namespace bmia


#endif // bmia_Core_h
