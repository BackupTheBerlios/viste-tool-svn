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
 * Settings_GeneralPageWidget.cxx
 *
 * 2011-07-13	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "Settings_GeneralPageWidget.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

Settings_GeneralPageWidget::Settings_GeneralPageWidget()
{
	// Create radio buttons for the window size (maximized or custom size).
	QRadioButton * maximizedRadioButton = new QRadioButton("Maximized");
		maximizedRadioButton->setChecked(true);
	QRadioButton * customSizeRadioButton = new QRadioButton("Custom Size");
		maximizedRadioButton->setChecked(false);

	// Add the custom size radio button to a horizontal layout, along with two 
	// spin boxes for setting the desired width and height (and a label and
	// spacer for layout purposes).

	QSpacerItem * customSizeSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
	QSpinBox * customSizeWSpin = new QSpinBox;
		customSizeWSpin->setMinimum(800);
		customSizeWSpin->setMaximum(9999);
		customSizeWSpin->setValue(800);
		customSizeWSpin->setEnabled(false);
		customSizeWSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	QLabel * customSizeXLabel = new QLabel("X");
		customSizeXLabel->setEnabled(false);
	QSpinBox * customSizeHSpin = new QSpinBox;
		customSizeHSpin->setMinimum(600);
		customSizeHSpin->setMaximum(10000);
		customSizeHSpin->setValue(600);
		customSizeHSpin->setEnabled(false);
		customSizeHSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	QHBoxLayout * customSizeHLayout = new QHBoxLayout;
		customSizeHLayout->addWidget(customSizeRadioButton);
		customSizeHLayout->addSpacerItem(customSizeSpacer);
		customSizeHLayout->addWidget(customSizeWSpin);
		customSizeHLayout->addWidget(customSizeXLabel);
		customSizeHLayout->addWidget(customSizeHSpin);

	// Create a vertical layout containing the maximize radio button and the custom size layout
	QVBoxLayout * windowSizeVLayout = new QVBoxLayout;
		windowSizeVLayout->addWidget(maximizedRadioButton);
		windowSizeVLayout->addLayout(customSizeHLayout);

	// Add this layout to a new group box
	QGroupBox * windowSizeGroup = new QGroupBox("Window Size");
		windowSizeGroup->setLayout(windowSizeVLayout);

	// Create a frame for the background color, and a button for changing this color
	QFrame * bgColorFrame = new QFrame;
		bgColorFrame->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		bgColorFrame->setFixedSize(40, 20);
		bgColorFrame->setFrameShape(QFrame::NoFrame);
		bgColorFrame->setFrameShadow(QFrame::Plain);
	QPushButton * bgColorButton = new QPushButton("Pick Color...");

	// Add the frame and the button to a horizontal layout
	QHBoxLayout * bgColorHLayout = new QHBoxLayout;
		bgColorHLayout->addWidget(bgColorFrame);
		bgColorHLayout->addWidget(bgColorButton, 1);

	// Create a check box for the background gradient
	QCheckBox * bgColorGradientCheck = new QCheckBox("Background Gradient");

	// Add all three background color controls to a single group
	QVBoxLayout * bgColorVLayout = new QVBoxLayout;
		bgColorVLayout->addLayout(bgColorHLayout);
		bgColorVLayout->addWidget(bgColorGradientCheck);
	QGroupBox * bgColorGroup = new QGroupBox("Background Color");
		bgColorGroup->setLayout(bgColorVLayout);

	// This page has been divided into two columns. Add the window size group to
	// the left column, along with a spacer.

	QSpacerItem * leftColumnSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Expanding);
	QVBoxLayout * leftColumnVLayout = new QVBoxLayout;
		leftColumnVLayout->addWidget(windowSizeGroup);
		leftColumnVLayout->addSpacerItem(leftColumnSpacer);

	// The right column contains the background color group
	QSpacerItem * rightColumnSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Expanding);
	QVBoxLayout * rightColumnVLayout = new QVBoxLayout;
		rightColumnVLayout->addWidget(bgColorGroup);
		rightColumnVLayout->addSpacerItem(rightColumnSpacer);

	// Add the columns to a horizontal layout
	QHBoxLayout * pageLayout = new QHBoxLayout;
		pageLayout->addLayout(leftColumnVLayout);
		pageLayout->addLayout(rightColumnVLayout);

	// Use this horizontal layout as the main layout for this page
	this->mainLayout = (QLayout *) pageLayout;
	this->setLayout(this->mainLayout);

	// Disable the custom size spin boxes when the maximized button is checked
	connect(customSizeRadioButton, SIGNAL(toggled(bool)), customSizeWSpin,  SLOT(setEnabled(bool)));
	connect(customSizeRadioButton, SIGNAL(toggled(bool)), customSizeHSpin,  SLOT(setEnabled(bool)));
	connect(customSizeRadioButton, SIGNAL(toggled(bool)), customSizeXLabel, SLOT(setEnabled(bool)));

	// Launch a color dialog when the user clicks the "Pick Color" button
	connect(bgColorButton, SIGNAL(clicked()), this, SLOT(pickColor()));

	// Set "settingsModified" to true whenever one of the active controls is used
	connect(customSizeRadioButton, SIGNAL(toggled(bool)),     this, SLOT(setSettingsModified()));
	connect(maximizedRadioButton,  SIGNAL(toggled(bool)),     this, SLOT(setSettingsModified()));
	connect(customSizeWSpin,       SIGNAL(valueChanged(int)), this, SLOT(setSettingsModified()));
	connect(customSizeHSpin,       SIGNAL(valueChanged(int)), this, SLOT(setSettingsModified()));
	connect(bgColorGradientCheck,  SIGNAL(toggled(bool)),     this, SLOT(setSettingsModified()));

	// Store the pointers of the active controls for easy access later on
	this->pMaximizeWindowRadio	= maximizedRadioButton;
	this->pCustomSizeRadio		= customSizeRadioButton;
	this->pWindowWidthSpin		= customSizeWSpin;
	this->pWindowHeightSpin		= customSizeHSpin;
	this->pBGColorFrame			= bgColorFrame;
	this->pBGGradientCheck		= bgColorGradientCheck;
}


//------------------------------[ Destructor ]-----------------------------\\

Settings_GeneralPageWidget::~Settings_GeneralPageWidget()
{
	// Nothing to do here; the main layout is destroyed in the parent class
}


//-----------------------------[ getPageName ]-----------------------------\\

QString Settings_GeneralPageWidget::getPageName()
{
	return "General Settings";
}


//--------------------------[ initializeControls ]-------------------------\\

void Settings_GeneralPageWidget::initializeControls(DTIToolSettings * settings)
{
	// New settings, so they have not yet been modified
	this->settingsModified = false;

	// Initialize settings for the window size
	this->pMaximizeWindowRadio->setChecked(settings->maximizeWindow);
	this->pCustomSizeRadio->setChecked(!(settings->maximizeWindow));
	this->pWindowWidthSpin->setValue(settings->windowWidth);
	this->pWindowHeightSpin->setValue(settings->windowHeight);

	// Change the color of the background color frame
	QPalette palette = this->pBGColorFrame->palette();
	palette.setColor(this->pBGColorFrame->backgroundRole(), settings->backgroundColor);
	this->pBGColorFrame->setPalette(palette);
	this->pBGColorFrame->setAutoFillBackground(true);

	// Check or uncheck the gradient checkbox
	this->pBGGradientCheck->setChecked(settings->gradientBackground);
}


//----------------------------[ storeSettings ]----------------------------\\

bool Settings_GeneralPageWidget::storeSettings(DTIToolSettings * settings)
{
	// Do nothing if no settings were modified
	if (this->settingsModified == false)
		return false;

	// Store the settings for the window size
	settings->maximizeWindow = this->pMaximizeWindowRadio->isChecked();
	settings->windowWidth    = this->pWindowWidthSpin->value();
	settings->windowHeight   = this->pWindowHeightSpin->value();

	// Store the settings for the background color
	settings->backgroundColor = this->pBGColorFrame->palette().color(this->pBGColorFrame->backgroundRole());
	settings->gradientBackground = this->pBGGradientCheck->isChecked();

	// Modified settings have been stored, so the controls are now up-to-date
	this->settingsModified = false;

	// Return true to signal that we've changed the settings
	return true;
}


//------------------------------[ pickColor ]------------------------------\\

void Settings_GeneralPageWidget::pickColor()
{
	// Get the old color from the color frame
	QColor oldColor = this->pBGColorFrame->palette().color(this->pBGColorFrame->backgroundRole());

	// Use a color dialog to pick a new color
	QColor newColor = QColorDialog::getColor(oldColor, this, "Pick background color...");

	// Make sure the color is valid
	if (!(newColor.isValid()))
		return;

	// The settings on this page have now been modified
	this->setSettingsModified();

	// Change the color of the frame to the new color
	QPalette palette = this->pBGColorFrame->palette();
	palette.setColor(this->pBGColorFrame->backgroundRole(), newColor);
	this->pBGColorFrame->setPalette(palette);
	this->pBGColorFrame->setAutoFillBackground(true);
}


} // namespace bmia
