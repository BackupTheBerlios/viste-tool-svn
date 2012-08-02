/*
 * IllustrativeClustersPlugin.cxx
 *
 * 2011-03-24	Evert van Aart
 * - Version 1.0.0.
 * - First version. Based on the work by Ron Otten from the old tool, but adapted
 *   to the new plugin system and data management system.
 *
 * 2011-07-12	Evert van Aart
 * - Version 1.0.1.
 * - User can now show/hide fibers and clusters from the GUI.
 *
 */


#include "IllustrativeClustersPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

IllustrativeClustersPlugin::IllustrativeClustersPlugin() : AdvancedPlugin("Illustrative Clusters")
{
	// Create the GUI
	this->widget = new QWidget();
	this->ui = new Ui::IllustrativeClustersForm();
	this->ui->setupUi(this->widget);

	// Create the assembly, which holds all ROI actors
	this->assembly = vtkPropAssembly::New();

	// Add some pre-defined pairs of line- and fill colors to the list. These
	// colors are stored as a hexadecimal string with two digits per color (RGB),
	// prefaced by a "#". This is the exact format required by the "setNamedColor"
	// function of "QColor".

	this->lineColorList.append("#815154");		this->fillColorList.append("#FFA0A5");
	this->lineColorList.append("#504E87");		this->fillColorList.append("#9793FF");
	this->lineColorList.append("#49834D");		this->fillColorList.append("#8EFF96");
	this->lineColorList.append("#828247");		this->fillColorList.append("#FEFF8C");
	this->lineColorList.append("#7F6542");		this->fillColorList.append("#FFCC85");
	this->lineColorList.append("#488B89");		this->fillColorList.append("#85FFFB");
	this->lineColorList.append("#9A6A6A");		this->fillColorList.append("#EFCFCF");
	this->lineColorList.append("#914989");		this->fillColorList.append("#FF81FE");

	// Index used for getting colors from the two color lists
	this->colorListIndex = 0;

	// Connect controls for showing and hiding fibers and clusters
	connect(this->ui->showClusterButton, SIGNAL(clicked()), this, SLOT(showCluster()));
	connect(this->ui->hideClusterButton, SIGNAL(clicked()), this, SLOT(hideCluster()));
	connect(this->ui->showFibersButton, SIGNAL(clicked()), this, SLOT(showFibers()));
	connect(this->ui->hideFibersButton, SIGNAL(clicked()), this, SLOT(hideFibers()));

	// Connect displacement controls
	connect(this->ui->dispEnableCheck, SIGNAL(toggled(bool)), this, SLOT(toggleDisplacement(bool)));
	connect(this->ui->focusWidgetCheck, SIGNAL(toggled(bool)), this, SLOT(toggleDisplacementWidget(bool)));
	connect(this->ui->dispExplosionSlide, SIGNAL(valueChanged(int)), this, SLOT(updateDisplacementOptions()));
	connect(this->ui->dispSlideSlide, SIGNAL(valueChanged(int)), this, SLOT(updateDisplacementOptions()));
	connect(this->ui->focusRegionButton, SIGNAL(clicked()), this, SLOT(focusOnWidget()));
	connect(this->ui->focusClusterButton, SIGNAL(clicked()), this, SLOT(focusOnCluster()));

	// Connect color picking controls
	connect(this->ui->colorLineChangeButton, SIGNAL(clicked()), this, SLOT(changeLineColor()));
	connect(this->ui->colorFillChangeButton, SIGNAL(clicked()), this, SLOT(changeFillColor()));
	connect(this->ui->colorLineCopyButton, SIGNAL(clicked()), this, SLOT(copyFillColorToLine()));
	connect(this->ui->colorFillCopyButton, SIGNAL(clicked()), this, SLOT(copyLineColorToFill()));

	// Connect the "Current Cluster" combo box
	connect(this->ui->currentClusterCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(copyConfigToGUI()));

	// Connect the "Apply" buttons
	connect(this->ui->updateButton, SIGNAL(clicked()), this, SLOT(applyToCurrent()));
	connect(this->ui->updateAllButton, SIGNAL(clicked()), this, SLOT(applyToAll()));

	// Create a timer, and connect it to the animation function
	this->mTimerAnimation = new QTimer(this);
	this->mTimerAnimation->setInterval(20);
	connect(this->mTimerAnimation, SIGNAL(timeout()), this, SLOT(animateDisplacement()));
	mTimerAnimation->start();
}


//---------------------------------[ init ]--------------------------------\\

void IllustrativeClustersPlugin::init()
{
	// Create the displacement manager
	this->displacement = new IllustrativeClusterDisplacement(this->fullCore()->canvas());
}


//------------------------------[ Destructor ]-----------------------------\\

IllustrativeClustersPlugin::~IllustrativeClustersPlugin()
{
	// Clear the string lists
	this->fillColorList.clear();
	this->lineColorList.clear();

	// Clear the list of input data sets
	this->inDataSets.clear();

	// Delete all clusters
	for (int i = 0; i < this->clusterList.size(); ++i)
	{
		this->deleteCluster(i);
	}

	// Clear the list of cluster pointers
	this->clusterList.clear();

	// Delete the GUI widget
	delete this->widget;

	// Delete the assembly object
	if (this->assembly)
		this->assembly->Delete();
}


//----------------------------[ deleteCluster ]----------------------------\\

void IllustrativeClustersPlugin::deleteCluster(int clusterID)
{
	if (clusterID < 0 || clusterID >= this->clusterList.size())
		return;

	// Get the target cluster
	IllustrativeCluster * cluster = this->clusterList.at(clusterID);

	// Get the actor
	vtkActor * actor = cluster->getActor();

	// Remove the actor from the assembly
	this->assembly->RemovePart(actor);

	// Delete the cluster itself
	delete cluster;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * IllustrativeClustersPlugin::getGUI()
{
	return this->widget;
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp * IllustrativeClustersPlugin::getVtkProp()
{
	return this->assembly;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void IllustrativeClustersPlugin::dataSetAdded(bmia::data::DataSet * ds)
{
	// We can only use fiber data sets
	if (ds->getKind() != "fibers" || this->inDataSets.contains(ds))
		return;

	// Get the polydata of the fibers
	vtkSmartPointer<vtkPolyData> polyData = ds->getVtkPolyData();

	if (!polyData)
		return;

	// Create a new cluster object, with a new actor and a new mapper
	IllustrativeCluster * newCluster = new IllustrativeCluster(ds->getName(),
																vtkSmartPointer<vtkActor>::New(),
																vtkSmartPointer<vtkIllustrativeFiberBundleMapper>::New());

	// Get a color from the list of pre-defined colors
	int loopedColorListIndex = this->colorListIndex % this->lineColorList.size();
	newCluster->SetColors(	this->lineColorList.at(loopedColorListIndex),
							this->fillColorList.at(loopedColorListIndex));
	this->colorListIndex++;

	// Setup the mapper and actor
	newCluster->getMapper()->SetInput(polyData);
	newCluster->getActor()->SetMapper(newCluster->getMapper());

	// Copy the default configuration to the mapper
	this->updateMapperSettings(newCluster->getMapper(), newCluster->getConfiguration());	

	// Add the actor to the assembly
	this->assembly->AddPart(newCluster->getActor());

	// Force the mapper to update now
	newCluster->getMapper()->Update();	

	// Add the input fiber set to the list of input data sets
	this->inDataSets.append(ds);

	// Add the new cluster to the list of clusters
	this->clusterList.append(newCluster);

	// Add the cluster name to the combo box. This will also trigger the "copyConfigToGUI"
	// function, which will set the GUI widget values according to the default configuration.

	this->ui->currentClusterCombo->addItem(ds->getName());

	// Add the cluster actor to the displacement manager, and update it
	this->displacement->addActor(newCluster->getActor());
	this->displacement->updateInput();

	// Hide the input fibers by default
	ds->getAttributes()->addAttribute("isVisible", -1.0);
	this->core()->data()->dataSetChanged(ds);

	// Done, let's render the new scene
	this->core()->render();
}


//----------------------------[ dataSetChanged ]---------------------------\\

void IllustrativeClustersPlugin::dataSetChanged(bmia::data::DataSet * ds)
{
	// Since the "vtkPolyData" object is hooked up the illustrative cluster
	// mapper, changes to the fibers will be correctly propagated through this
	// plugin. We therefore only have to worry about two things: Changes in the
	// "vtkPolyData" pointer, and name changes.

	// We're only interested in fibers
	if (ds->getKind() != "fibers")
		return;

	// Check if the data set has been added to this plugin
	if (!(this->inDataSets.contains(ds)))
		return;

	// Get the index of the data set
	int dsIndex = this->inDataSets.indexOf(ds);

	// Get the corresponding cluster
	IllustrativeCluster * currentCluster = this->clusterList.at(dsIndex);

	// If necessary, change the input of the mapper, and re-draw the scene
	if (currentCluster->getMapper()->GetInput() != ds->getVtkPolyData())
	{
		currentCluster->getMapper()->SetInput(ds->getVtkPolyData());
		currentCluster->getMapper()->Update();
		this->core()->render();
	}

	// Otherwise, just update the name of the cluster
	currentCluster->updateName(ds->getName());
	this->ui->currentClusterCombo->setItemText(dsIndex, ds->getName());
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void IllustrativeClustersPlugin::dataSetRemoved(bmia::data::DataSet * ds)
{
	// We're only interested in fibers
	if (ds->getKind() != "fibers")
		return;

	// Check if the data set has been added to this plugin
	if (!(this->inDataSets.contains(ds)))
		return;

	// Get the index of the data set
	int dsIndex = this->inDataSets.indexOf(ds);

	// Remove the actor from the displacement manager
	IllustrativeCluster * cluster = this->clusterList.at(dsIndex);
	this->displacement->removeActor(cluster->getActor());

	// Delete the cluster
	this->deleteCluster(dsIndex);

	// Remove the items from the lists and the GUI
	this->inDataSets.removeAt(dsIndex);
	this->clusterList.removeAt(dsIndex);
	this->ui->currentClusterCombo->removeItem(dsIndex);
}


//-----------------------------[ hideCluster ]-----------------------------\\

void IllustrativeClustersPlugin::hideCluster()
{
	// Get the current cluster
	int clusterID = this->ui->currentClusterCombo->currentIndex();

	if (clusterID < 0 || clusterID >= this->inDataSets.size())
		return;

	IllustrativeCluster * cluster = this->clusterList.at(clusterID);

	if (!cluster)
		return;

	// Hide the cluster
	cluster->getActor()->SetVisibility(0);
}


//-----------------------------[ showCluster ]-----------------------------\\

void IllustrativeClustersPlugin::showCluster()
{
	// Get the current cluster
	int clusterID = this->ui->currentClusterCombo->currentIndex();

	if (clusterID < 0 || clusterID >= this->inDataSets.size())
		return;

	IllustrativeCluster * cluster = this->clusterList.at(clusterID);

	if (!cluster)
		return;

	// Show the cluster
	cluster->getActor()->SetVisibility(1);
}


//------------------------------[ showFibers ]-----------------------------\\

void IllustrativeClustersPlugin::showFibers()
{
	// Get the input fiber data set
	int clusterID = this->ui->currentClusterCombo->currentIndex();

	if (clusterID < 0 || clusterID >= this->inDataSets.size())
		return;

	data::DataSet * inDS = this->inDataSets[clusterID];

	// Set the visibility attribute, and signal the data manager about the change
	inDS->getAttributes()->addAttribute("isVisible", 1.0);
	this->core()->data()->dataSetChanged(inDS);
}


//------------------------------[ hideFibers ]-----------------------------\\

void IllustrativeClustersPlugin::hideFibers()
{
	// Get the input fiber data set
	int clusterID = this->ui->currentClusterCombo->currentIndex();

	if (clusterID < 0 || clusterID >= this->inDataSets.size())
		return;

	data::DataSet * inDS = this->inDataSets[clusterID];

	// Set the visibility attribute, and signal the data manager about the change
	inDS->getAttributes()->addAttribute("isVisible", -1.0);
	this->core()->data()->dataSetChanged(inDS);
}


//------------------------[ updateMapperSettings ]-------------------------\\

void IllustrativeClustersPlugin::updateMapperSettings(int clusterID)
{
	// Check if the index is within range
	if (clusterID < 0 || clusterID >= this->clusterList.size())
		return;

	// Get the cluster object
	IllustrativeCluster * cluster = this->clusterList.at(clusterID);

	// Get the mapper and the configuration, and call the next function
	this->updateMapperSettings(cluster->getMapper(), cluster->getConfiguration());
}


//------------------------[ updateMapperSettings ]-------------------------\\

void IllustrativeClustersPlugin::updateMapperSettings(vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mapper,
													  const IllustrativeCluster::Configuration & configuration)
{
	// Copy all configuration settings to the mapper
	mapper->SetLineColor(configuration.lineColor.redF(), configuration.lineColor.greenF(), configuration.lineColor.blueF());
	mapper->SetFillColor(configuration.fillColor.redF(), configuration.fillColor.greenF(), configuration.fillColor.blueF());
	mapper->SetLighting(configuration.phongConstants.x, configuration.phongConstants.y, configuration.phongConstants.z);
	mapper->SetShinyness(configuration.specularPower);
	mapper->SetFinWidth(configuration.haloWidth);
	mapper->SetFinRecision(configuration.haloDepth);
	mapper->SetMinimumStrokeWidth(configuration.minStrokeWidth);
	mapper->SetMaximumStrokeWidth(configuration.maxStrokeWidth);
	mapper->SetMinimumLuminosity(configuration.minLuminosity);
	mapper->SetMaximumLuminosity(configuration.maxLuminosity);
	mapper->CreateSilhouette(configuration.enableSilhouette);
	mapper->SetFillDilation(configuration.silhouetteWidth);
	mapper->SetOutlineWidth(configuration.contourWidth);
	mapper->SetInnerOutlineDepthThreshold(configuration.depthThreshold);
	mapper->UseStroking(configuration.enableCurvatureStrokes);
	mapper->ApplyLightingToLines(configuration.enableLighting);
}


//--------------------------[ toggleDisplacement ]-------------------------\\

void IllustrativeClustersPlugin::toggleDisplacement(bool enable)
{
	// Enable or disable the displacement manager, and update it
	this->displacement->setIsActive(enable);
	this->displacement->updateInput();

	// Enable or disable GUI controls
	this->ui->dispExplosionLabel->setEnabled(enable);
	this->ui->dispExplosionSpin->setEnabled(enable);
	this->ui->dispExplosionSlide->setEnabled(enable);
	this->ui->dispSlideLabel->setEnabled(enable);
	this->ui->dispSlideSpin->setEnabled(enable);
	this->ui->dispSlideSlide->setEnabled(enable);
}


//-----------------------[ toggleDisplacementWidget ]----------------------\\

void IllustrativeClustersPlugin::toggleDisplacementWidget(bool enable)
{
	// Enable or disable the focus widget
	this->displacement->enableFocusSelectionWidget(enable);
}


//----------------------[ updateDisplacementOptions ]----------------------\\

void IllustrativeClustersPlugin::updateDisplacementOptions()
{
	// Update the scale parameters
	float newExplosionScale = (float) this->ui->dispExplosionSlide->value() / 100.0f;
	float newSlideScale =     (float) this->ui->dispSlideSlide->value()     / 100.0f;
	this->displacement->setScales(newExplosionScale, newSlideScale);

	// Update the displacement manager
	this->displacement->updateInput();
}


//----------------------------[ focusOnWidget ]----------------------------\\

void IllustrativeClustersPlugin::focusOnWidget()
{
	this->displacement->setFocusToSelection();
	this->displacement->updateInput();
}


//----------------------------[ focusOnCluster ]---------------------------\\

void IllustrativeClustersPlugin::focusOnCluster()
{
	// Get the current cluster object, and focus on it
	int currentClusterID = this->ui->currentClusterCombo->currentIndex();
	data::DataSet * currentClusterDS = this->inDataSets.at(currentClusterID);
	this->displacement->setFocusToCurrentCluster(currentClusterDS->getVtkPolyData());
	this->displacement->updateInput();
}


//---------------------------[ changeLineColor ]---------------------------\\

void IllustrativeClustersPlugin::changeLineColor()
{
	// Get a new color, and set it to the line color preview frame
	QColor oldColor = this->getFrameColor(this->ui->colorLineFrame);
	QColor newColor = QColorDialog::getColor(oldColor, this->widget, "Choose new line color...");
	this->setFrameColor(this->ui->colorLineFrame, newColor);
}


//---------------------------[ changeFillColor ]---------------------------\\

void IllustrativeClustersPlugin::changeFillColor()
{
	// Get a new color, and set it to the fill color preview frame
	QColor oldColor = this->getFrameColor(this->ui->colorFillFrame);
	QColor newColor = QColorDialog::getColor(oldColor, this->widget, "Choose new fill color...");
	this->setFrameColor(this->ui->colorFillFrame, newColor);
}


//-------------------------[ copyFillColorToLine ]-------------------------\\

void IllustrativeClustersPlugin::copyFillColorToLine()
{
	this->setFrameColor(this->ui->colorLineFrame, this->getFrameColor(this->ui->colorFillFrame));
}


//-------------------------[ copyLineColorToFill ]-------------------------\\

void IllustrativeClustersPlugin::copyLineColorToFill()
{
	this->setFrameColor(this->ui->colorFillFrame, this->getFrameColor(this->ui->colorLineFrame));
}


//----------------------------[ setFrameColor ]----------------------------\\

void IllustrativeClustersPlugin::setFrameColor(QFrame * colorFrame, QColor newColor)
{
	if (newColor.isValid())
	{
		// Set the new color as the background color for the frame
		QPalette palette = colorFrame->palette();
		palette.setColor(colorFrame->backgroundRole(), newColor);
		colorFrame->setPalette(palette);
	}
}


//----------------------------[ getFrameColor ]----------------------------\\

QColor IllustrativeClustersPlugin::getFrameColor(QFrame * colorFrame)
{
	// Get the background color of the frame
	return colorFrame->palette().color(colorFrame->backgroundRole());
}


//---------------------------[ copyConfigToGUI ]---------------------------\\

void IllustrativeClustersPlugin::copyConfigToGUI()
{
	// Get the current cluster object and its configuration object
	int currentClusterID = this->ui->currentClusterCombo->currentIndex();

	IllustrativeCluster * currentCluster = this->clusterList.at(currentClusterID);

	IllustrativeCluster::Configuration config = currentCluster->getConfiguration();

	// Copy the configuration parameters to the GUI widgets
	this->setFrameColor(this->ui->colorLineFrame, config.lineColor);
	this->setFrameColor(this->ui->colorFillFrame, config.fillColor);

	this->ui->haloWidthSlide->setValue((int) (config.haloWidth * 10.0f + 0.5f));
	this->ui->haloWidthSpin->setValue((int) (config.haloWidth * 10.0f + 0.5f));
	this->ui->haloDepthSlide->setValue((int) (config.haloDepth * 10.0f + 0.5f));
	this->ui->haloDepthSpin->setValue((int) (config.haloDepth * 10.0f + 0.5f));

	this->ui->strokeGroup->setChecked(config.enableCurvatureStrokes);
	this->ui->strokeMinSlide->setValue((int) (config.minStrokeWidth * 100.0f + 0.5f));
	this->ui->strokeMinSpin->setValue((int) (config.minStrokeWidth * 100.0f + 0.5f));
	this->ui->strokeMaxSlide->setValue((int) (config.maxStrokeWidth * 100.0f + 0.5f));
	this->ui->strokeMaxSpin->setValue((int) (config.maxStrokeWidth * 100.0f + 0.5f));

	this->ui->lightEnableCheck->setChecked(config.enableLighting);
	this->ui->lightAmbientSlide->setValue((int) (config.phongConstants.x * 100.0f + 0.5f));
	this->ui->lightAmbientSpin->setValue((int) (config.phongConstants.x * 100.0f + 0.5f));
	this->ui->lightDiffuseSlide->setValue((int) (config.phongConstants.y * 100.0f + 0.5f));
	this->ui->lightDiffuseSpin->setValue((int) (config.phongConstants.y * 100.0f + 0.5f));
	this->ui->lightSpecularSlide->setValue((int) (config.phongConstants.z * 100.0f + 0.5f));
	this->ui->lightSpecularSpin->setValue((int) (config.phongConstants.z * 100.0f + 0.5f));
	this->ui->lightSpecularPowerSpin->setValue((int) (config.specularPower));

	this->ui->lightClampMinSlide->setValue((int) (config.minLuminosity * 100.0f + 0.5f));
	this->ui->lightClampMinSpin->setValue((int) (config.minLuminosity * 100.0f + 0.5f));
	this->ui->lightClampMaxSlide->setValue((int) (config.maxLuminosity * 100.0f + 0.5f));
	this->ui->lightClampMaxSpin->setValue((int) (config.maxLuminosity * 100.0f + 0.5f));

	this->ui->silhGroup->setChecked(config.enableSilhouette);
	this->ui->silhThresholdSlide->setValue(config.depthThreshold);
	this->ui->silhThresholdSpin->setValue(config.depthThreshold);
	this->ui->silhWidthSlide->setValue(config.silhouetteWidth);
	this->ui->silhWidthSpin->setValue(config.silhouetteWidth);
	this->ui->silhContourSlide->setValue(config.contourWidth);
	this->ui->silhContourSpin->setValue(config.contourWidth);
}


//---------------------------[ copyGUItoConfig ]---------------------------\\

void IllustrativeClustersPlugin::copyGUItoConfig(IllustrativeCluster * cluster, bool copyColors)
{
	// Get the old configuration, and create a new one
	IllustrativeCluster::Configuration oldConfig = cluster->getConfiguration();
	IllustrativeCluster::Configuration newConfig;

	// For the current cluster, we use the colors of the two preview frames
	if (copyColors)
	{
		newConfig.lineColor = this->getFrameColor(this->ui->colorLineFrame);
		newConfig.fillColor = this->getFrameColor(this->ui->colorFillFrame);
	}
	// For all other clusters, we use the current color. This is used when clicking
	// the "Apply to All" button, since we do not want to set the same color to all
	// clusters.

	else
	{
		newConfig.lineColor = oldConfig.lineColor;
		newConfig.fillColor = oldConfig.fillColor;
	}

	// Copy the rest of the values of the GUI widgets to the new configuration
	newConfig.haloWidth = (float) this->ui->haloWidthSlide->value() / 10.0f;
	newConfig.haloDepth = (float) this->ui->haloDepthSlide->value() / 10.0f;

	newConfig.enableCurvatureStrokes = this->ui->strokeGroup->isChecked();
	newConfig.minStrokeWidth = (float) this->ui->strokeMinSpin->value() / 100.0f;
	newConfig.maxStrokeWidth = (float) this->ui->strokeMaxSpin->value() / 100.0f;

	newConfig.enableLighting = this->ui->lightEnableCheck->isChecked();
	newConfig.phongConstants.x = (float) this->ui->lightAmbientSpin->value() / 100.0f;
	newConfig.phongConstants.y = (float) this->ui->lightDiffuseSpin->value() / 100.0f;
	newConfig.phongConstants.z = (float) this->ui->lightSpecularSpin->value() / 100.0f;
	newConfig.specularPower = this->ui->lightSpecularPowerSpin->value();

	newConfig.minLuminosity = (float) this->ui->lightClampMinSpin->value() / 100.0f;
	newConfig.maxLuminosity = (float) this->ui->lightClampMaxSpin->value() / 100.0f;

	newConfig.enableSilhouette = this->ui->silhGroup->isChecked();
	newConfig.contourWidth = this->ui->silhContourSpin->value();
	newConfig.depthThreshold = this->ui->silhThresholdSpin->value();
	newConfig.silhouetteWidth = this->ui->silhWidthSpin->value();

	// Store the new configuration
	cluster->updateConfiguration(newConfig);
}


//----------------------------[ applyToCurrent ]---------------------------\\

void IllustrativeClustersPlugin::applyToCurrent()
{
	// Get the current cluster
	int currentClusterID = this->ui->currentClusterCombo->currentIndex();
	IllustrativeCluster * currentCluster = this->clusterList.at(currentClusterID);

	// copy the GUI values to its configuration
	this->copyGUItoConfig(currentCluster, true);

	// Update its mapper with these new settings
	this->updateMapperSettings(currentClusterID);

	// Render the scene to see the changes
	this->core()->render();
}


//------------------------------[ applyToAll ]-----------------------------\\

void IllustrativeClustersPlugin::applyToAll()
{
	// Get the index of the current cluster
	int currentClusterID = this->ui->currentClusterCombo->currentIndex();

	IllustrativeCluster * currentCluster = NULL;

	// Loop through all clusters
	for (int i = 0; i < this->clusterList.size(); ++i)
	{
		// Get the current cluster
		currentCluster = this->clusterList.at(i);

		// Copy the GUI widget values to the cluster's configuration. We only copy
		// the line- and fill colors if this is the selected cluster.

		this->copyGUItoConfig(currentCluster, (i == currentClusterID));

		// Update the cluster's mapper
		this->updateMapperSettings(i);
	}

	// Done, render the scene
	this->core()->render();
}


//-------------------------[ animateDisplacement ]-------------------------\\

void IllustrativeClustersPlugin::animateDisplacement()
{
	this->displacement->updateAnimation(20);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libIllustrativeClustersPlugin, bmia::IllustrativeClustersPlugin)
