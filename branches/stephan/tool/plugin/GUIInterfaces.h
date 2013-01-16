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
 * VisualizationInterface.h
 *
 * 2010-02-19	Wiljan van Ravensteijn
 * - First version
 *
 */


#ifndef bmia_plugin_GUIInterfaces_h
#define bmia_plugin_GUIInterfaces_h


class QWidget;


namespace bmia {


namespace plugin {


/** This class implements plugins with GUI functionality. All plugins with 
	this functionality should implement the function "getGUI", which returns the 
	Qt Widget that defines the Graphical User Interface of a plugin.  
*/

class GUI
{
	public:
		
		/** Destructor. */
    
		virtual ~GUI() 
		{

		};

		/** Returns the Qt widget that will be added to the interface. */

		virtual QWidget * getGUI() = 0;

	protected:

	private:

}; // class GUI


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::GUI, "bmia.plugin.GUI")


#endif // bmia_plugin_GUIInterfaces_h
