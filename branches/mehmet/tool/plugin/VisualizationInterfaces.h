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
 * 2010-02-11	Tim Peeters
 * - First version.
 *
 */


#ifndef bmia_plugin_VisualizationInterfaces_h
#define bmia_plugin_VisualizationInterfaces_h


class vtkProp;


namespace bmia {


namespace plugin {


/** This class implements plugins with visualization functionality. Plugins with
	this functionality should implement the "getVtkProp" function, which returns
	a VTK prop object. This function is called once when the plugin is created. 
	The returned prop is then added to the list of props of the renderer, which 
	will display it on the screen. The visualization should also create and configure
	the mapper that will be used to draw the prop. If you need to draw more than one
	VTK prop with one visualization plugin, you can add them to a "vtkPropAssembly"
	object, and return this assembly in the "getVtkProp" function.
*/

class Visualization
{
	public:
    
		/** Destructor */

		virtual ~Visualization() 
		{

		};

		/** Return a VTK Prop that will be added to the tool's main renderer when 
			this visualization plugin is loaded. The function must always return 
			the same pointer, which may not be NULL. */

		virtual vtkProp * getVtkProp() = 0;

	protected:

	private:

}; // class Visualization


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::Visualization, "bmia.plugin.Visualization")


#endif // bmia_plugin_VisualizationInterfaces_h
