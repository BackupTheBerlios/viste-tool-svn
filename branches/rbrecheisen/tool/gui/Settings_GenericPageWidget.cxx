/*
 * Settings_GenericPageWidget.cxx
 *
 * 2011-07-13	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "Settings_GenericPageWidget.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

Settings_GenericPageWidget::Settings_GenericPageWidget()
{
	// Set pointer to NULL
	this->mainLayout = NULL;

	// At the start, the settings have not yet been modified
	this->settingsModified = false;
}


//------------------------------[ Destructor ]-----------------------------\\

Settings_GenericPageWidget::~Settings_GenericPageWidget()
{
	// Delete the main layout. This will also delete all child widgets, layouts, and spacers
	if (this->mainLayout)
		delete this->mainLayout;
}


} // namespace bmia
