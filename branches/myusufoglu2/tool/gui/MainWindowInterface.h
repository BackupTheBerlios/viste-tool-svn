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
 * MainWindowInterface.h
 *
 * 2010-08-05 Tim Peeters
 * - First version
 *
 */


#ifndef bmia_gui_MainWindowInterface_h
#define bmia_gui_MainWindowInterface_h


/** Includes - Qt */

#include <QWidget>
#include <QString>

/** Includes - VTK */

#include <vtkProp.h>

/** Includes - Custom Files */

#include "core/DTIToolSettings.h"


namespace bmia {


namespace gui {


/** Interface for the main window class, used to shield off vertain functions from
	classes that should not have full access to all public functions. 
*/

class MainWindowInterface
{
	public:
	
		/** Add a plugin GUI to the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */
	
		virtual void addPluginGui(QWidget * widget, QString name) = 0;

		/** Remove a plugin GUI from the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */

		virtual void removePluginGui(QWidget * widget, QString name) = 0;

		/** Add a VTK prop to the VTK renderer. 
			@param prop		VTK prop of a visualization plugin.
			@param name		Plugin name. */

		virtual void addPluginVtkProp(vtkProp * prop, QString name) = 0;

		/** Remove a VTK prop from the VTK renderer. 
			@param prop		VTK prop of a visualization plugin. */

		virtual void removePluginVtkProp(vtkProp * prop) = 0;

		/** Apply settings of the DTITool. This is called by the "applySettings"
			function of the core object, which first applies those settings that
			are directly related to the core. This function then applies the settings
			that are related to the main window (e.g., background color, window size). 
			@param settings	DTITool settings. */
	
		virtual void applySettings(DTIToolSettings * settings) = 0;

}; // class MainWindowInterface


} // namespace gui


} // namespace bmia


#endif // bmai_gui_MainWindowInterface_h
