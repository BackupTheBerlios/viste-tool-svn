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
