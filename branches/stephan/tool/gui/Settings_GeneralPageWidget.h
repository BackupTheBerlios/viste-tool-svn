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
 * Settings_GeneralPageWidget.h
 *
 * 2011-07-13	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_Settings_GeneralPageWidget_h
#define bmia_Settings_GeneralPageWidget_h


/** Includes - Custom Files */

#include "Settings_GenericPageWidget.h"

/** Includes - Qt */

#include <QGroupBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QLabel>
#include <QSpacerItem>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QColorDialog>
#include <QPushButton>
#include <QCheckBox>


namespace bmia {


/** This class represents a settings page containing general settings, like
	the window size. It inherits its core functionality from "Settings_GenericPageWidget".
*/

class Settings_GeneralPageWidget : public Settings_GenericPageWidget
{
	Q_OBJECT

	public:

		/** Constructor */

		Settings_GeneralPageWidget();

		/** Destructor */

		~Settings_GeneralPageWidget();

		/** Return the name of this page. */

		virtual QString getPageName();

		/** Copy current settings to GUI controls. 
			@param settings	Input DTITool settings. */

		virtual void initializeControls(DTIToolSettings * settings);

		/** Copy GUI control values back to the settings; return true if the settings
			were modified, and false otherwise. 
			@param settings	Output DTITool settings. */

		virtual bool storeSettings(DTIToolSettings * settings);

	protected:

		QRadioButton * pMaximizeWindowRadio;	/**< Radio button for maximizing the window. */
		QRadioButton * pCustomSizeRadio;		/**< Radio button for setting a custom window size. */
		QSpinBox * pWindowWidthSpin;			/**< Spinner for the window width. */
		QSpinBox * pWindowHeightSpin;			/**< Spinner for the window height. */
		QFrame * pBGColorFrame;					/**< Frame displaying the current background color. */
		QCheckBox * pBGGradientCheck;			/**< Check box for gradient background colors. */

	protected slots:

		/** Use a color dialog to pick a new background color. */

		void pickColor();

}; // class Settings_GeneralPageWidget



} // namespace bmia


#endif // bmia_Settings_GeneralPageWidget_h