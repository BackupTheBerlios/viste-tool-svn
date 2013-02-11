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
 * MainWindow.cxx
 *
 * 2009-12-22	Tim Peeters
 * - First version
 *
 * 2010-11-29	Evert van Aart
 * - Added support for writing profile file.
 *
 * 2011-01-03	Evert van Aart
 * - Added support for multiple filters (one filter per supported file type).
 * - Added a button for writing and reading the default folder.
 * - Added support for loading multiple files.
 *
 * 2011-02-28	Evert van Aart
 * - User can now change the background color and enable or disable the 
 *   background gradient.
 * - Added support for making screenshots.
 *
 * 2011-03-18	Evert van Aart
 * - Added the Profile Manager, which allows the user to create, modify and delete
 *   settings profile, as well as a dialog for choosing the default profile.
 * - Removed functions for the old settings method (i.e., those related to 
 *   "profile.ini" and "folder.ini", since these options are now all part of
 *   the new "settings.xml" system.
 *
 * 2011-03-25	Evert van Aart
 * - Screenshots can now also be PNGs and BMPs.
 *
 * 2011-06-23	Evert van Aart
 * - Window is now maximized by default.
 *
 * 2011-07-18	Evert van Aart
 * - Added the settings dialog, which can be used to change general settings.
 * - Moved the background color options to this settings dialog.
 * - Added comments, implemented destructor.
 *
 */


/** Includes */

#include "MainWindow.h"
#include "SettingsDialog.h"
#include "vtkMedicalCanvas.h"


namespace bmia {


namespace gui {


//-----------------------------[ Constructor ]-----------------------------\\

MainWindow::MainWindow(Core * coreInstance, QWidget * parent) : QMainWindow(parent)
{
	// Check for valid input
	Q_ASSERT(coreInstance);
	Q_ASSERT(coreInstance->plugin());
	Q_ASSERT(coreInstance->data());

	// Store the core pointer
	this->core = coreInstance;

	// Initialize the GUI of the main window
	this->setupUi(this);

	// Make the layouts for the frames that will show the plugin GUIs
	this->pluginFrameTop->setLayout(new QVBoxLayout);
	this->pluginFrameBot->setLayout(new QVBoxLayout);

	// Create the plugin and data dialogs
	this->pluginDialog = new PluginDialog(this->core->plugin(), this);
	this->dataDialog   = new DataDialog(  this->core->data(),   this);

	// Setup the renderer
	this->renderer		= NULL;
	this->metaCanvas	= NULL;
	this->initializeVtkRenderWindow();

	this->core->setMainWindow(this);
	this->core->setRenderWindow(this->vtkWidget->GetRenderWindow());
	this->core->setMedicalCanvas(this->metaCanvas);

	// Add the plugins toolbar
	this->pluginToolbar = this->addToolBar("Plugins");

	// Connect the menu actions to the correct functions
	this->connectActions();

	this->setTabPosition(Qt::AllDockWidgetAreas, QTabWidget::North);

	// Add a simple string as the empty GUI widget
	this->addPluginGui(new QLabel("Select a plugin"), "None");

	// Create a signal mapper for the visibility buttons on the toolbar
	this->visibilitySignalMapper = new QSignalMapper(this);
	connect(this->visibilitySignalMapper, SIGNAL(mapped(const QString &)), this, SIGNAL(visToggled(const QString &)));
	connect(this, SIGNAL(visToggled(const QString&)), this, SLOT(showVis(const QString&)));

	// Connect the plugin chooser combo boxes
	connect(this->pluginChooserTop, SIGNAL(currentIndexChanged(int)), this, SLOT(selectPluginGui()));
	connect(this->pluginChooserBot, SIGNAL(currentIndexChanged(int)), this, SLOT(selectPluginGui()));

	// Minimum window size
	this->setMinimumSize(800, 600);

	// Clear all shortcut actions
	for (int i = 0; i < 10; ++i)
	{
		this->guiShortcutActions[i] = NULL;
	}

	this->guiShortcutMapper = NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

MainWindow::~MainWindow()
{
	// Renderer is deleted when we delete the canvas
	this->renderer = NULL;

	// Core is deleted separately
	this->core = NULL;

	// Delete the plugin and data dialogs
	if (this->pluginDialog)
		delete this->pluginDialog;

	if (this->dataDialog)
		delete this->dataDialog;

	// Delete the canvas
	this->metaCanvas->Delete();

	// Clear assorted lists. We do not delete the widgets and VTK props here,
	// the plugins should take care of that when they're deleted.

	this->pluginWidgets.clear();
	this->pluginNames.clear();
	this->vtkProps.clear();
	this->vtkPropsNames.clear();
	
	// Delete active GUI shortcuts
	for (int i = 0; i < 10; ++i)
	{
		if (this->guiShortcutActions[i])
			delete this->guiShortcutActions[i];
	}

	// Delete the visibility actions, and clear their list
	for (QList<QAction *>::iterator i = this->toggleVisActions.begin(); i != this->toggleVisActions.end(); ++i)
	{
		if (*i)
			delete (*i);
	}

	this->toggleVisActions.clear();

	// Delete the toolbar
	if (this->pluginToolbar)
		delete this->pluginToolbar;

	// Delete the signal mappers
	if (this->guiShortcutMapper)
		delete this->guiShortcutMapper;

	if (this->visibilitySignalMapper)
		delete this->visibilitySignalMapper;
}


//----------------------------[ connectActions ]---------------------------\\

void MainWindow::connectActions()
{
	// Connect the actions in the main menu
    connect(this->actionOpenData,				SIGNAL(triggered()), this, SLOT(openData()));
    connect(this->actionListData,				SIGNAL(triggered()), this, SLOT(listData()));
	connect(this->actionSaveData,				SIGNAL(triggered()), this, SLOT(saveData()));
    connect(this->actionQuit,					SIGNAL(triggered()), this, SLOT(quit()));
    connect(this->actionListPlugins,			SIGNAL(triggered()), this, SLOT(listPlugins()));
	connect(this->actionProfileManager,			SIGNAL(triggered()), this, SLOT(launchProfileManager()));
	connect(this->actionSettings,				SIGNAL(triggered()), this, SLOT(launchSettingsDialog()));
	connect(this->actionSet_Default_Profile,	SIGNAL(triggered()), this, SLOT(launchDefaultProfileDialog()));
	connect(this->actionAbout,					SIGNAL(triggered()), this, SLOT(showAbout()));
	connect(this->actionScreenshot,				SIGNAL(triggered()), this, SLOT(makeScreenshot()));
}


//----------------------------[ makeScreenshot ]---------------------------\\

void MainWindow::makeScreenshot()
{
	// Open a file dialog to get the target file name
	QString fileName = QFileDialog::getSaveFileName(this,
		"Save Screenshot as...",
		"",	
		"Portable Network Graphics (*.png);;JPEG Image (*.jpg);;Bitmap Image (*.bmp)");

	if (fileName.isEmpty() || fileName.isNull())
		return;

	// Create a window to image filter
	vtkWindowToImageFilter * windowToImageFilter = vtkWindowToImageFilter::New();

	// Set the input of the filter and update it
	windowToImageFilter->SetInput(this->getMetaCanvas()->GetRenderWindow());
	windowToImageFilter->Update();

	vtkImageWriter * imageWriter = NULL;

	// PNG
	if (fileName.endsWith(".png", Qt::CaseInsensitive))
	{
		vtkPNGWriter * writer = vtkPNGWriter::New();
		imageWriter = vtkImageWriter::SafeDownCast(writer);
	}
	// JPG
	else if (fileName.endsWith(".jpg", Qt::CaseInsensitive))
	{
		vtkJPEGWriter * writer = vtkJPEGWriter::New();

		// A quality of 80 is usually a nice trade-off between quality and file-size
		writer->SetQuality(80);
		imageWriter = vtkImageWriter::SafeDownCast(writer);
	}
	// BMP
	else if (fileName.endsWith(".bmp", Qt::CaseInsensitive))
	{
		vtkBMPWriter * writer = vtkBMPWriter::New();
		imageWriter = vtkImageWriter::SafeDownCast(writer);
	}
	// If we don't recognize the file extension, we use PNG by default
	else
	{
		fileName.append(".png");
		vtkPNGWriter * writer = vtkPNGWriter::New();
		imageWriter = vtkImageWriter::SafeDownCast(writer);
	}

	// Render the window now to make sure it's up-to-date
	this->getMetaCanvas()->GetRenderWindow()->Render();

	// Set the output of the filter as the input of the writer
	imageWriter->SetInput(windowToImageFilter->GetOutput());

	// Set the filename, and create the image
	imageWriter->SetFileName(fileName.toLatin1());
	imageWriter->Write();

	// Done, get rid of the VTK objects
	imageWriter->Delete();
	windowToImageFilter->Delete();
}


//---------------------------[ changeBackground ]--------------------------\\

void MainWindow::changeBackground(QColor newColor, bool applyGradient)
{
	// Get the RGB values as doubles
	double rgb[3];
	rgb[0] = (double) newColor.redF();
	rgb[1] = (double) newColor.greenF();
	rgb[2] = (double) newColor.blueF();

	// Gradient backgrounds
	if (applyGradient)
	{
		// Increment the RGB values to get the secondary color
		double rgb2[3];
		rgb2[0] = rgb[0] + 0.2;
		rgb2[1] = rgb[1] + 0.2;
		rgb2[2] = rgb[2] + 0.2; 

		// Clamp the RGB values to 1.0
		rgb2[0] = (rgb2[0] > 1.0) ? (1.0) : (rgb2[0]);
		rgb2[1] = (rgb2[1] > 1.0) ? (1.0) : (rgb2[1]);
		rgb2[2] = (rgb2[2] > 1.0) ? (1.0) : (rgb2[2]);

		// Set the gradient background to all canvasses
		this->getMetaCanvas()->setGradientBackground(rgb[0], rgb[1], rgb[2], rgb2[0], rgb2[1], rgb2[2]);
	}
	// Uniform backgrounds
	else
	{
		// Set the uniform color to all canvasses
		this->getMetaCanvas()->setBackgroundColor(rgb[0], rgb[1], rgb[2]);
	}
}


//-------------------------------[ openData ]------------------------------\\

void MainWindow::openData()
{
	// String containing the default folder
	QString defaultFolder = "";

	// If we've got at least one profile, we can get the default folder
	if (this->core->profiles.size() > 0)
	{
		// Get the path of the data directory of the first profile, in case
		// no default profile has been set (should not happen).

		defaultFolder = this->core->profiles.at(0)->dataDir.absolutePath();

		// Loop through the remainder of the profile
		for (int i = 1; i < this->core->profiles.size(); ++i)
		{
			// Get the current profile
			DTIToolProfile * currentProfile = this->core->profiles.at(i);

			// Check if this is the default profile
			if (currentProfile->isDefault)
			{
				// If so, get the path of the profile's data directory
				defaultFolder = currentProfile->dataDir.absolutePath();
				break;
			}
		}
	}

	// Get the list of individual filter strings (format: "File Type (*.ext)")
	QStringList filterList = this->core->data()->getSupportedFileExtensionsWithDescriptions();

	// Do nothing if no file types are supported
	if (filterList.isEmpty())
		return;

	// Copy the first filter string to the output
	QString filterString = filterList.at(0);

	// Add the rest of the filter strings, separated by double semicolumns
	for (int i = 1; i < filterList.size(); ++i)
	{
		filterString.append(";;");
		filterString.append(filterList.at(i));
	}

	// Get filename of one or more file
    QStringList filenames = QFileDialog::getOpenFileNames(	this,				// Parent
															"Choose file(s)",	// Caption
															defaultFolder,		// Directory
															filterString);		// Filter

	// Do nothing if the list of names is empty
    if (filenames.isEmpty()) 
		return;

	// Loop through all filenames
	foreach(QString filename, filenames)
	{
		// Do nothing if the filename is invalid
		if (filename.isEmpty() || filename.isNull())
			continue;

		// Load the file
		this->core->data()->loadDataFromFile(filename);
	}
}

//-------------------------------[ saveData ]------------------------------\\

void MainWindow::saveData()
{
	this->dataDialog->show();
}



//-------------------------------[ listData ]------------------------------\\

void MainWindow::listData()
{
	this->dataDialog->show();
}


//--------------------------[ refreshPluginList ]--------------------------\\

void MainWindow::refreshPluginList()
{
	this->pluginDialog->rebuildList();
}


//---------------------------------[ quit ]--------------------------------\\

void MainWindow::quit()
{
    this->close();
}


//-----------------------------[ listPlugins ]-----------------------------\\

void MainWindow::listPlugins()
{
	// String containing the default plugin folder
	QString defaultFolder = "";

	// If we've got at least one profile, we can get the default folder
	if (this->core->profiles.size() > 0)
	{
		// Get the path of the data directory of the first profile, in case
		// no default profile has been set (should not happen).

		defaultFolder = this->core->profiles.at(0)->pluginDir.absolutePath();

		// Loop through the remainder of the profile
		for (int i = 1; i < this->core->profiles.size(); ++i)
		{
			// Get the current profile
			DTIToolProfile * currentProfile = this->core->profiles.at(i);

			// Check if this is the default profile
			if (currentProfile->isDefault)
			{
				// If so, get the path of the profile's data directory
				defaultFolder = currentProfile->pluginDir.absolutePath();
				break;
			}
		}
	}

	this->pluginDialog->setPluginDir(defaultFolder);
	this->pluginDialog->show();
}


//-------------------------[ launchSettingsDialog ]------------------------\\

void MainWindow::launchSettingsDialog()
{
	// Create the settings dialog
	SettingsDialog * settingsDialog = new SettingsDialog(this->core->settings, this);
	settingsDialog->setFixedSize(800, 600);
	settingsDialog->exec();

	// Check if the dialog was accepted, and if changes were made to the settings
	if (settingsDialog->result() == (int) QDialog::Accepted && settingsDialog->settingsModified)
	{
		// If so, apply the settings now
		this->core->applySettings();
	}

	// Delete the dialog
	delete settingsDialog;
}


//----------------------------[ applySettings ]----------------------------\\

void MainWindow::applySettings(DTIToolSettings * settings)
{
	// Either maximize the window...
	if (settings->maximizeWindow)
	{
		this->showMaximized();
	}
	// ...or set it to a specified size
	else
	{
		this->showNormal();
		this->resize(settings->windowWidth, settings->windowHeight);
	}

	// Set the background color and gradient
	this->changeBackground(settings->backgroundColor, settings->gradientBackground);

	// Loop through all ten possible GUI shortcuts
	for (int i = 0; i < 10; ++i)
	{
		// Copy the shortcut information to this class
		this->guiShortcuts[i] = settings->guiShortcuts[i];

		// Delete existing actions
		if (this->guiShortcutActions[i])
		{
			this->removeAction(this->guiShortcutActions[i]);
			delete this->guiShortcutActions[i];
		}

		this->guiShortcutActions[i] = NULL;
	}

	// Delete existing shortcut mapper
	if (this->guiShortcutMapper)
		delete this->guiShortcutMapper;

	// Create a new signal mapper. This mapper allows us to map all shortcut actions
	// to one slot function ("guiShortcutPressed"), but with a different index, based
	// on which shortcut was used (e.g., if the user presses "Ctrl+2", the input 
	// argument of "guiShortcutPressed" will be two).

	this->guiShortcutMapper = new QSignalMapper;

	// Loop through all ten possible GUI shortcuts
	for (int i = 0; i < 10; ++i)
	{
		// If the plugin isn't "None", and the plugin is currently loaded...
		if (settings->guiShortcuts[i].plugin != "None" &&
			this->pluginChooserTop->findText(settings->guiShortcuts[i].plugin) != -1)
		{
			// ...create a shortcut action for this shortcut
			this->guiShortcutActions[i] = new QAction(this);
			QString shortcutString = "Ctrl+" + QString::number(i);
			this->guiShortcutActions[i]->setShortcut(QKeySequence(shortcutString));
			this->guiShortcutActions[i]->setShortcutContext(Qt::ApplicationShortcut);

			// Add a mapping from the new action to its index to the signal mapper
			this->guiShortcutMapper->setMapping(this->guiShortcutActions[i], i);

			// Connect this action to the mapper
			connect(this->guiShortcutActions[i], SIGNAL(triggered()), this->guiShortcutMapper, SLOT(map()));

			// Add the action to the window
			this->addAction(this->guiShortcutActions[i]);
		}
	}

	// Connect the output of the mapper to the "guiShortcutPressed" function
	connect(this->guiShortcutMapper, SIGNAL(mapped(int)), this, SLOT(guiShortcutPressed(int)));
}


//--------------------------[ guiShortcutPressed ]-------------------------\\

void MainWindow::guiShortcutPressed(int key)
{
	// Get the name of the target plugin
	QString targetPlugin = this->guiShortcuts[key].plugin;

	// Find the index of the plugin
	int pluginIndex = this->pluginChooserTop->findText(targetPlugin);

	// Do nothing if the index is out of range
	if (pluginIndex < 0 || pluginIndex > 9)
		return;

	// Get the target position of the GUI (top or bottom)
	DTIToolSettings::GUIPosition position = this->guiShortcuts[key].position;

	// Select the plugin in the top combo box
	if (position == DTIToolSettings::GUIP_Top)
	{
		this->pluginChooserTop->setCurrentIndex(pluginIndex);
	}

	// Select the plugin in the bottom combo box
	else if (position == DTIToolSettings::GUIP_Bottom)
	{
		this->pluginChooserBot->setCurrentIndex(pluginIndex);
	}

	// Select the plugin in the top combo box, and clear the bottom combo box
	else if (position == DTIToolSettings::GUIP_TopExclusive)
	{
		this->pluginChooserBot->setCurrentIndex(0);
		this->pluginChooserTop->setCurrentIndex(pluginIndex);
	}
}


//-------------------------[ launchProfileManager ]------------------------\\

void MainWindow::launchProfileManager()
{
	// Create the profile manager and launch it
	ProfileManager * manager = new ProfileManager(this->core->plugin(), this->core->settings);
	manager->setFixedSize(600, 800);
	manager->addProfileList(&(this->core->profiles));
	manager->exec();

	bool defaultProfileSet = false;

	// Check if any of the profiles has "isDefault" set to true
	for (int i = 0; i < this->core->profiles.size(); ++i)
	{
		if (this->core->profiles.at(i)->isDefault)
		{
			defaultProfileSet = true;
			break;
		}
	}

	// If not, force the user to choose a default profile now (usually happens
	// if the user has deleted the default profile in the manager).

	if (defaultProfileSet == false)
	{
		QMessageBox::information(NULL, "Set default profile", "No default profile set, please select one now.");

		this->launchDefaultProfileDialog();
	}

	// Done!
	delete manager;
}


//----------------------[ launchDefaultProfileDialog ]---------------------\\

void MainWindow::launchDefaultProfileDialog()
{
	// Create the dialog and launch it
	DefaultProfileDialog * dialog = new DefaultProfileDialog(&(this->core->profiles), this->core->settings);
	dialog->setFixedSize(250, 250);
	dialog->exec();

	// Inform the user of our success
	QMessageBox::information(NULL, "Default profile set", "Default profile has been set. The new default profile will be loaded when the tool is restarted.");

	delete dialog;
}


//----------------------[ initializeVtkRenderWindow ]----------------------\\

void MainWindow::initializeVtkRenderWindow()
{
	// Renderer and meta-canvas have not yet been created
	Q_ASSERT(this->renderer == NULL);
	Q_ASSERT(this->metaCanvas == NULL);

	// Create the meta-canvas and add it to this window
	this->metaCanvas = vtkMedicalCanvas::New();
	this->metaCanvas->SetRenderWindow(this->vtkWidget->GetRenderWindow());
	this->metaCanvas->SetInteractor(this->vtkWidget->GetInteractor());

	// Store the 3D renderer of the meta-canvas
	this->renderer = this->metaCanvas->GetRenderer3D();
}


//----------------------------[ getVtkRenderer ]---------------------------\\

vtkRenderer * MainWindow::getVtkRenderer()
{
	Q_ASSERT(this->renderer);
	return this->renderer;
}


//----------------------------[ getMetaCanvas ]----------------------------\\

vtkMedicalCanvas * MainWindow::getMetaCanvas()
{
	Q_ASSERT(this->metaCanvas);
	return this->metaCanvas;
}


//------------------------------[ vtkRender ]------------------------------\\

void MainWindow::vtkRender()
{
	this->vtkWidget->GetRenderWindow()->Render();
}


//-----------------------------[ addPluginGui ]----------------------------\\

void MainWindow::addPluginGui(QWidget * widget, QString name)
{
	// Add the name and widget pointer to lists
	this->pluginNames.append(name);
	this->pluginWidgets.append(widget);

	// Add the name to the two plugin chooser combo boxes
	this->pluginChooserTop->addItem(name);
	this->pluginChooserBot->addItem(name);
}


//---------------------------[ removePluginGui ]---------------------------\\

void MainWindow::removePluginGui(QWidget * widget, QString name)
{
	// Get the index of the removed plugin
	int i = this->pluginNames.indexOf(name);

	if (i < 0)
		return;

	// Remove the plugin name from the combo boxes
	this->pluginChooserTop->removeItem(i);
	this->pluginChooserBot->removeItem(i);

	// Remove the name and the widget pointer from the lists
	this->pluginNames.removeAt(i);
	this->pluginWidgets.removeAt(i);

	// Reselect plugin GUIs based on the new combo box contents
	this->selectPluginGui();
}


//---------------------------[ selectPluginGui ]---------------------------\\

void MainWindow::selectPluginGui()
{
	// Get the index of the top and bottom plugin choosers
	int indexTop = this->pluginChooserTop->currentIndex();
	int indexBot = this->pluginChooserBot->currentIndex();

	// If the same plugin has been selected in both boxes, clear the bottom one
	if (indexTop == indexBot) 
		indexBot = 0;

	// Make sure the indices are in the correct range
	if (indexBot < 0 || indexBot >= this->pluginNames.size() || 
		indexTop < 0 || indexTop >= this->pluginNames.size() )
		return;

	// Loop through all plugins
	for (int i = 0; i < this->pluginWidgets.size(); i++)
	{
		// Get the current plugin widget, and hide it
		QWidget * widget = this->pluginWidgets.at(i);
	
		Q_ASSERT(widget);
	
		widget->hide();

		// Remove the plugin widget from the top and the bottom GUI fields
		this->pluginFrameTop->layout()->removeWidget(widget);
		this->pluginFrameBot->layout()->removeWidget(widget);
	}

	// Add the selected plugin widgets to the top and bottom fields
	this->pluginFrameTop->layout()->addWidget(this->pluginWidgets.at(indexTop));
	this->pluginWidgets.at(indexTop)->show();

	this->pluginFrameBot->layout()->addWidget(this->pluginWidgets.at(indexBot));
	this->pluginWidgets.at(indexBot)->show();
}


//---------------------------[ addPluginVtkProp ]--------------------------\\

void MainWindow::addPluginVtkProp(vtkProp * prop, QString name)
{
	// Create a new VTK prop for turning visualization plugins on or off
	QAction * toggleVisAction = new QAction(name, this);
	toggleVisAction->setCheckable(true);
	toggleVisAction->setChecked(true);

	// Add this action to the toolbar as a button
	this->pluginToolbar->addAction(toggleVisAction);

	// Map the action to the plugin name
	this->visibilitySignalMapper->setMapping(toggleVisAction, name);
	connect(toggleVisAction, SIGNAL(toggled(bool)), this->visibilitySignalMapper, SLOT(map()));

	// Store the action, the prop pointer, and the prop name
	this->toggleVisActions.push_back(toggleVisAction);
	this->vtkProps.push_back(prop);
	this->vtkPropsNames.push_back(name);

	// Add the prop to the renderer
	this->getVtkRenderer()->AddViewProp(prop);
	this->vtkRender();
}


//-------------------------[ removePluginVtkProp ]-------------------------\\

void MainWindow::removePluginVtkProp(vtkProp * prop)
{
	// Find the prop in the list
	int i = this->vtkProps.indexOf(prop);

	// Make sure the index is in the correct range
	if (i < 0 || i >= this->vtkProps.size())
		return;

	// Remove the prop from the renderer
	this->getVtkRenderer()->RemoveViewProp(prop);

	// Delete the action. This automatically removes plugin from the toolbar, and
	// removes the connections in signal mapper.
	delete this->toggleVisActions.at(i);

	// Remove the action information from the lists
	this->toggleVisActions.removeAt(i);
	this->vtkProps.removeAt(i);
	this->vtkPropsNames.removeAt(i);    

	// Render the scene
	this->vtkRender();
}


//-------------------------------[ showVis ]-------------------------------\\

void MainWindow::showVis(const QString &pluginName)
{
	// Find the index of the plugin
	int i = this->vtkPropsNames.indexOf(pluginName);

	if (i < 0 || i >= this->vtkPropsNames.size())
		return;

	// Get the prop for this plugin
	vtkProp * prop = this->vtkProps.at(i);

	Q_ASSERT(this->renderer);
	Q_ASSERT(prop);

	// Either remove the prop from the renderer...
	if (this->getVtkRenderer()->HasViewProp(prop))
	{
		this->getVtkRenderer()->RemoveViewProp(prop);
	}
	// ...or add it (again).
	else
	{
		this->getVtkRenderer()->AddViewProp(prop);
	}

	// Render the scene
	this->vtkRender();
}


//------------------------------[ showAbout ]------------------------------\\

void MainWindow::showAbout()
{
	// Create the credits string
	QString acks = QString("The following people contributed to the DTITool3 software:\n\n") +
	QString("Tim Peeters\n") +
	QString("Anna Vilanova\n") +
	QString("Evert van Aart\n") +
	QString("Paulo Rodrigues\n") +
	QString("Vesna Prchkovska\n") +
	QString("Ralph Brecheisen\n") +
	QString("Wiljan van Ravensteijn\n")+
	QString("Guus Berenschot\n") +
	QString("\n") +
	QString("To get the latest updates, and more information,\n") +
	QString("visit our website:\n\n") +
	QString("http://bmia.bmt.tue.nl/Software/DTITool/");
	;

	// Show the credits string in a message box
	QMessageBox box;
	box.setText("DTITool 3 Acknowledgments");
	box.setInformativeText(acks);
	box.exec();
}


} // namespace gui


} // namespace bmia
