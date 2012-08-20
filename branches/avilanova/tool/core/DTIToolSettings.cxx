/*
 * DTIToolSettings.cxx
 *
 * 2011-07-18	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "DTIToolSettings.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

DTIToolSettings::DTIToolSettings()
{
	// Initialize the settings
	this->setDefaultSettings();
}


//------------------------------[ Destructor ]-----------------------------\\

DTIToolSettings::~DTIToolSettings()
{

}


//--------------------------[ setDefaultSettings ]-------------------------\\

void DTIToolSettings::setDefaultSettings()
{
	// Maximize window by default
	this->windowWidth  = 800;
	this->windowHeight = 600;
	this->maximizeWindow = true;

	// Use a black background with a gradient
	this->backgroundColor = QColor(0, 0, 0);
	this->gradientBackground = true;

	// Disable all shortcuts
	for (int i = 0; i < 10; ++i)
	{
		this->guiShortcuts[i].plugin = "None";
		this->guiShortcuts[i].position = GUIP_Top;
	}
}


} // namespace bmia
