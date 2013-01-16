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
 * SimpleCoreInterface
 *
 * 2010-05-04	Tim Peeters	
 * - First version
 *
 * 2011-04-06	Evert van Aart
 * - Added "getDataDirectory".
 *
 * 2011-04-14	Evert van Aart
 * - Added "disableRendering" and "enableRendering".
 *
 */


#ifndef bmia_SimpleCoreInterface_h
#define bmia_SimpleCoreInterface_h


/** Includes - Qt */

#include <QDir>

namespace bmia {


class UserOutput;


namespace data 
{ 
	class Manager; 
}


/** Interface for the plugins to access the core. Used to control which functions
	can be accessed by regular plugins. */

class SimpleCoreInterface
{
	public:
    
		/** Return the data manager. */
    
		virtual data::Manager * data() = 0;

		/** Return the user output object. */
    
		virtual UserOutput * out() = 0;

		/** Call this function to draw the scene after making updates in the settings or pipeline. */
    
		virtual void render() = 0;

		/** Return the data directory. Reader plugins can use this to set the default directory
			of the file dialog. */

		virtual QDir getDataDirectory() = 0;

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

		virtual void disableRendering() = 0;

		/** Turn on rendering. See notes for "disableRendering". */

		virtual void enableRendering() = 0;

}; // class SimpleCoreInterface


} // namespace bmia


#endif // bmia_SimpleCoreInterface_h
