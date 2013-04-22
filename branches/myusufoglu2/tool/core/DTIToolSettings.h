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
 * DTIToolSettings.h
 *
 * 2011-07-18	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_DTIToolSettings_h
#define bmia_DTIToolSettings_h


/** Includes - Qt */

#include <QColor>


namespace bmia {


/** Simple class used to store the general settings of the DTITool. See the "Core"
	and "MainWindow" classes to see how these settings are applied. */

class DTIToolSettings
{
	public:

		/** Constructor */

		DTIToolSettings();

		/** Destructor */

		~DTIToolSettings();

		/** Initializes settings to their default values. */

		void setDefaultSettings();

		/** Position of the GUI, used for the GUI shortcuts. */

		enum GUIPosition
		{
			GUIP_Top = 0,			/**< Top field. */
			GUIP_TopExclusive,		/**< Top field (clear bottom field. */
			GUIP_Bottom				/**< Bottom field. */
		};

		/** Structure containing information a GUI shortcut. */

		struct GUIShortcutInfo
		{
			QString plugin;			/**< Plugin name. "None" if shortcut is disabled. */
			GUIPosition position;	/**< Position of the GUI (top or bottom field). */
		};

		int windowHeight;			/**< Window height in pixels. */
		int windowWidth;			/**< Window width in pixels. */
		bool maximizeWindow;		/**< Window is maximized if true. */

		QColor backgroundColor;		/**< Color of the background. */
		bool gradientBackground;	/**< Apply a gradient to the background. */

		/** Array of up to ten shortcuts for the GUI. */

		GUIShortcutInfo guiShortcuts[10];

}; // class DTIToolSettings


} // namespace bmia


#endif // bmia_DTIToolSettings_h
