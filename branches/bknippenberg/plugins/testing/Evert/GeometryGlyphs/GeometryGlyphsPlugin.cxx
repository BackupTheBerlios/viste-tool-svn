/*
 * GeometryGlyphsPlugin.cxx
 *
 * 2011-04-20	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 * 2011-05-09	Evert van Aart
 * - Version 1.1.0.
 * - Added additional support for coloring the glyphs.
 * - Added a glyph builder for Spherical Harmonics data.
 *
 */


/** Includes */

#include "GeometryGlyphsPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

GeometryGlyphsPlugin::GeometryGlyphsPlugin() : Plugin("Geometry Glyphs")
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::GeometryGlyphsForm();
	this->ui->setupUi(this->widget);

	// Builder will be created when the first data set is added
	this->builder = NULL;

	// Create a smoothing filter, and set the default options
	this->smoothFilter = vtkSmoothPolyDataFilter::New();
	this->smoothFilter->SetNumberOfIterations(100);
	this->smoothFilter->SetFeatureEdgeSmoothing(0);
	this->smoothFilter->BoundarySmoothingOn();

	// Create a mapper, link it to the output of the glyph builder, and create an actor
	this->actor = vtkActor::New();
	this->mapper = vtkPolyDataMapper::New();

	// Set the default LUT
	this->setLUT(0);

	// Connect the GUI controls
	connect(this->ui->glyphDataCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(inputDataChanged(int))			);
	connect(this->ui->seedPointsCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(seedDataChanged(int))			);
	connect(this->ui->normMethodCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(setNormalizationMethod(int))		);
	connect(this->ui->normScopeCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(setNormalizationScope(int))		);
	connect(this->ui->glyphTypeCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(setGlyphType(int))				);
	connect(this->ui->colorMethodCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeColorMethod(int))			);
	connect(this->ui->colorScalarsCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(setScalarVolume(int))			);
	connect(this->ui->colorLUTCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(setLUT(int))						);
	connect(this->ui->scaleSpin,			SIGNAL(valueChanged(double)),		this, SLOT(setScale(double))				);
	connect(this->ui->sharpenPowerSpin,		SIGNAL(valueChanged(double)),		this, SLOT(setSharpeningExponent(double))	);
	connect(this->ui->normalizeGroup,		SIGNAL(toggled(bool)),				this, SLOT(enableNormalization(bool))		);
	connect(this->ui->sharpenGroup,			SIGNAL(toggled(bool)),				this, SLOT(enableSharpening(bool))			);
	connect(this->ui->smoothGroup,			SIGNAL(toggled(bool)),				this, SLOT(enableSmoothing(bool))			);
	connect(this->ui->smoothUpdateButton,	SIGNAL(clicked()),					this, SLOT(updateSmoothOptions())			);
	connect(this->ui->tessSpin,				SIGNAL(valueChanged(int)),			this, SLOT(setTessellationOrder(int))		);
}


//---------------------------------[ init ]--------------------------------\\

void GeometryGlyphsPlugin::init()
{
	// Make sure that the progress is reported for the smoothing filter
	this->core()->out()->createProgressBarForAlgorithm(this->smoothFilter, "Geometry Glyphs", "Smoothing geometry glyphs...");
}


//------------------------------[ Destructor ]-----------------------------\\

GeometryGlyphsPlugin::~GeometryGlyphsPlugin()
{
	// Unload the GUI
	delete this->widget; 

	// Delete the actor
	this->actor->Delete();

	// Stop reporting the progress of the smoothing filter and the builder
	this->core()->out()->deleteProgressBarForAlgorithm(this->smoothFilter);
	this->core()->out()->deleteProgressBarForAlgorithm(this->builder);

	// Clear the input data set lists
	this->glyphDataSets.clear();
	this->seedDataSets.clear();
	this->scalarDataSets.clear();

	// Delete the visualization pipeline
	if (this->smoothFilter)
		this->smoothFilter->Delete();

	if (this->builder)
		this->builder->Delete();
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp * GeometryGlyphsPlugin::getVtkProp()
{
	return this->actor;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * GeometryGlyphsPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void GeometryGlyphsPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete sphere functions
	if (ds->getKind() == "discrete sphere" && this->glyphDataSets.contains(ds) == false)
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// Check if the point data contains a spherical directions array
		if (!(imagePD->GetArray("Spherical Directions")))
			return;

		// We can use this data set, so add it to the list and the GUI
		this->glyphDataSets.append(ds);
		this->ui->glyphDataCombo->addItem(ds->getName());
	}

	// Spherical Harmonics
	if (ds->getKind() == "spherical harmonics" && this->glyphDataSets.contains(ds) == false)
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data with scalars
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		if (!(imagePD->GetScalars()))
			return;

		// We can use this data set, so add it to the list and the GUI
		this->glyphDataSets.append(ds);
		this->ui->glyphDataCombo->addItem(ds->getName());
	}

	// Seed points
	else if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds) == false)
	{
		// Check if the data set contains a VTK data object
		if (!(ds->getVtkObject()))
			return;

		// If so, add it to the list and the GUI
		this->seedDataSets.append(ds);
		this->ui->seedPointsCombo->addItem(ds->getName());
	}

	// Scalar volumes
	else if (ds->getKind() == "scalar volume" && this->scalarDataSets.contains(ds) == false)
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// We can use this data set, so add it to the list and the GUI
		this->scalarDataSets.append(ds);
		this->ui->colorScalarsCombo->addItem(ds->getName());
	}

	// Transfer Function
	else if (ds->getKind() == "transfer function" && this->lutDataSets.contains(ds) == false)
	{
		if (!(ds->getVtkObject()))
			return;

		// Add the LUT data set to the lists
		this->lutDataSets.append(ds);
		this->ui->colorLUTCombo->addItem(ds->getName());
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void GeometryGlyphsPlugin::dataSetChanged(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions and Spherical Harmonics
	if ((ds->getKind() == "discrete sphere" || ds->getKind() == "spherical harmonics") && this->glyphDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->glyphDataSets.indexOf(ds);

		// Change the data set name
		this->ui->glyphDataCombo->setItemText(dsIndex, ds->getName());

		// If we're changing the currently selected data set...
		if (this->ui->glyphDataCombo->currentIndex() == dsIndex && this->builder)
		{
			// ...update the geometry of the builder, and render the scene
			this->builder->setInputVolume(this->glyphDataSets[dsIndex]->getVtkImageData());
			this->builder->computeGeometry(this->ui->tessSpin->value());
			this->builder->Modified();
			this->core()->render();
		}
	}

	// Seed points
	else if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->seedDataSets.indexOf(ds);

		// Change the data set name
		this->ui->seedPointsCombo->setItemText(dsIndex, ds->getName());

		// If we're changing the currently selected data set...
		if (this->ui->seedPointsCombo->currentIndex() == dsIndex && this->builder)
		{
			// ...update the builder, and render the scene
			this->builder->SetInput(0, vtkDataObject::SafeDownCast(this->seedDataSets[dsIndex]->getVtkObject()));
			this->builder->Modified();
			this->core()->render();
		}
	}

	// Scalar volumes
	else if (ds->getKind() == "scalar volume" && this->scalarDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->scalarDataSets.indexOf(ds);

		// Change the data set name
		this->ui->colorScalarsCombo->setItemText(dsIndex + 1, ds->getName());

		// If we're changing the currently selected data set...
		if ((this->ui->colorScalarsCombo->currentIndex() - 1) == dsIndex && this->builder)
		{
			// ...update the scalar volume pointer of the builder
			this->builder->setScalarVolume(ds->getVtkImageData());
			this->builder->Modified();
			this->core()->render();
		}
	}

	// Transfer Functions
	else if (ds->getKind() == "transfer function" && this->lutDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->lutDataSets.indexOf(ds);

		// Change the data set name
		this->ui->colorLUTCombo->setItemText(dsIndex + 1, ds->getName());

		// Check if we changed the currently selected LUT
		if (((this->ui->colorLUTCombo->currentIndex() - 1) == dsIndex && this->mapper))
		{
			// If so, update the LUT
			this->mapper->SetLookupTable(vtkScalarsToColors::SafeDownCast(ds->getVtkObject()));
			this->mapper->Modified();
			this->core()->render();
		}
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void GeometryGlyphsPlugin::dataSetRemoved(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions and Spherical Harmonics
	if ((ds->getKind() == "discrete sphere" || ds->getKind() == "spherical harmonics") && this->glyphDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->glyphDataSets.indexOf(ds);

		// Remove the data set from the list
		this->glyphDataSets.removeAt(dsIndex);

		if (dsIndex == this->ui->glyphDataCombo->currentIndex() && this->glyphDataSets.isEmpty())
		{
			// If we're deleting the last data set, first set the input of the builder to NULL
			if (this->builder)
				this->builder->setInputVolume(NULL);

			this->ui->glyphDataCombo->removeItem(dsIndex);
		}
		else
		{
			// Otherwise, just delete the item from the GUI, and let the "inputDataChanged"
			// function take care of the rest.

			this->ui->glyphDataCombo->removeItem(dsIndex);
		}
	}

	// Seed points
	else if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->seedDataSets.indexOf(ds);

		// Remove the data set from the list
		this->seedDataSets.removeAt(dsIndex);

		if (dsIndex == this->ui->seedPointsCombo->currentIndex() && this->seedDataSets.isEmpty())
		{
			// If we're deleting the last data set, first set the input of the builder to NULL
			if (this->builder)
				this->builder->SetInput(NULL);

			this->ui->seedPointsCombo->removeItem(dsIndex);
		}
		else
		{
			// Otherwise, just delete the item from the GUI, and let the "seedDataChanged"
			// function take care of the rest.

			this->ui->seedPointsCombo->removeItem(dsIndex);
		}
	}

	// Scalar volumes
	else if (ds->getKind() == "scalar volume" && this->scalarDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->scalarDataSets.indexOf(ds);

		// If we're changing the currently selected data set...
		if ((this->ui->colorScalarsCombo->currentIndex() - 1) == dsIndex && this->builder)
		{
			// ...switch to direction-based coloring
			this->ui->colorMethodCombo->setCurrentIndex(0);
			this->ui->colorScalarsCombo->setCurrentIndex(0);
			this->builder->Modified();
			this->builder->Update();
			this->core()->render();
		}

		// Remove the data set from the lists
		this->scalarDataSets.removeAt(dsIndex);
		this->ui->colorScalarsCombo->removeItem(dsIndex + 1);
	}

	// Transfer Function
	else if (ds->getKind() == "transfer function" && this->lutDataSets.contains(ds))
	{
		// Get the index of the data set
		int dsIndex = this->lutDataSets.indexOf(ds);

		// Check if we deleted the currently selected LUT
		if ((this->ui->colorLUTCombo->currentIndex() - 1) == dsIndex)
		{
			// If so, select the default LUT
			this->ui->colorLUTCombo->setCurrentIndex(0);
		}

		// Remove the data set from the lists
		this->lutDataSets.removeAt(dsIndex);
		this->ui->colorLUTCombo->removeItem(dsIndex + 1);
	}
}


//--------------------------[ inputDataChanged ]--------------------------\\

void GeometryGlyphsPlugin::inputDataChanged(int index)
{
	if (index < 0 || index >= this->glyphDataSets.size() || this->mapper == NULL)
		return;

	// Delete existing builder
	if (this->builder)
	{
		this->core()->out()->deleteProgressBarForAlgorithm(this->builder);
		this->builder->Delete();
	}

	// Create a builder for discrete sphere functions
	if (this->glyphDataSets[index]->getKind() == "discrete sphere")
	{
		// Disable the controls for tessellation, and enable the normalization scope combo box
		this->ui->tessLabel->setEnabled(false);
		this->ui->tessSpin->setEnabled(false);
		this->ui->normScopeCombo->setEnabled(true);

		this->builder = vtkGeometryGlyphBuilder::New();
	}

	// Create a builder for spherical harmonics
	else if (this->glyphDataSets[index]->getKind() == "spherical harmonics")
	{
		// Enable the controls for tessellation
		this->ui->tessLabel->setEnabled(true);
		this->ui->tessSpin->setEnabled(true);

		// Set the normalization scope to local
		disconnect(this->ui->normScopeCombo, SIGNAL(currentIndexChanged(int)),	this, SLOT(setNormalizationScope(int)));
		this->ui->normScopeCombo->setCurrentIndex(vtkGeometryGlyphBuilder::NS_Local);
		this->ui->normScopeCombo->setEnabled(false);
		connect(this->ui->normScopeCombo, SIGNAL(currentIndexChanged(int)),	this, SLOT(setNormalizationScope(int)));

		this->builder = (vtkGeometryGlyphBuilder *) vtkGeometryGlyphFromSHBuilder::New();
	}

	// Make sure that the progress is reported for the builder
	this->core()->out()->createProgressBarForAlgorithm(this->builder, "Geometry Glyphs");

	// True if we should use the identity matrix
	bool useIdentityMatrix = true;

	// Try to get a transformation matrix from the data set
	vtkObject * obj;
	if ((this->glyphDataSets[index]->getAttributes()->getAttribute("transformation matrix", obj)))
	{
		// Try to cast the object to a matrix
		if (vtkMatrix4x4::SafeDownCast(obj))
		{
			useIdentityMatrix = false;

			// Copy the matrix to a new one, and apply it to the actor
			vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);
			vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
			mCopy->DeepCopy(m);
			this->actor->SetUserMatrix(mCopy);
			mCopy->Delete();
		}
	}

	// If we didn't find a transformation matrix, apply the identity matrix
	if (useIdentityMatrix)
	{
		vtkMatrix4x4 * id = vtkMatrix4x4::New();
		id->Identity();
		this->actor->SetUserMatrix(id);
		id->Delete();
	}

	// Set the discrete sphere function volume, and compute its geometry template
	this->builder->setInputVolume(this->glyphDataSets[index]->getVtkImageData());
	this->builder->computeGeometry(this->ui->tessSpin->value());

	// Setup the pipeline
	this->builder->SetInput(0, vtkDataObject::SafeDownCast(this->seedDataSets[this->ui->seedPointsCombo->currentIndex()]->getVtkObject()));
	this->mapper->SetInput(this->builder->GetOutput());
	this->actor->SetMapper(this->mapper);
	this->setLUT(this->ui->colorLUTCombo->currentIndex());

	// Try to change the glyph type. If this fails (usually because the "3D Mesh"
	// type has been selected for a data set without triangulation), we switch back
	// to start-shaped glyphs. Error reporting in this case is handled by the
	// "setGlyphType" function.
	
	if (this->builder->setGlyphType((vtkGeometryGlyphBuilder::GeometryGlyphType) this->ui->glyphTypeCombo->currentIndex()) == false)
	{
		this->ui->glyphTypeCombo->setCurrentIndex(vtkGeometryGlyphBuilder::GGT_Star);
	}

	// Update the builder and render the scene
	this->core()->render();
}


//---------------------------[ seedDataChanged ]---------------------------\\

void GeometryGlyphsPlugin::seedDataChanged(int index)
{
	if (index < 0 || index >= this->seedDataSets.size())
		return;

	if (!this->builder)
		return;

	this->builder->SetInput(0, vtkDataObject::SafeDownCast(this->seedDataSets[index]->getVtkObject()));
	this->builder->Modified();
	this->builder->Update();

	this->core()->render();
}


//------------------------[ setNormalizationMethod ]-----------------------\\

void GeometryGlyphsPlugin::setNormalizationMethod(int index)
{
	if (this->builder == NULL)
		return;

	if (index == 0)
		this->builder->setNormalizationMethod(vtkGeometryGlyphBuilder::NM_MinMax);
	else if (index == 1)
		this->builder->setNormalizationMethod(vtkGeometryGlyphBuilder::NM_Maximum);

	this->builder->Modified();
	this->core()->render();
}


//------------------------[ setNormalizationScope ]------------------------\\

void GeometryGlyphsPlugin::setNormalizationScope(int index)
{
	if (this->builder == NULL)
		return;

	if (index == 0)
		this->builder->setNormalizationScope(vtkGeometryGlyphBuilder::NS_WholeImage);
	else if (index == 1)
		this->builder->setNormalizationScope(vtkGeometryGlyphBuilder::NS_SeedPoints);
	else if (index == 2)
		this->builder->setNormalizationScope(vtkGeometryGlyphBuilder::NS_Local);

	this->builder->Modified();
	this->core()->render();
}


//-------------------------------[ setScale ]------------------------------\\

void GeometryGlyphsPlugin::setScale(double scale)
{
	if (this->builder == NULL)
		return;

	this->builder->setScale(scale);
	this->builder->Modified();
	this->core()->render();
}


//------------------------[ setSharpeningExponent ]------------------------\\

void GeometryGlyphsPlugin::setSharpeningExponent(double exponent)
{
	if (this->builder == NULL)
		return;

	this->builder->setSharpeningExponent(exponent);
	this->builder->Modified();
	this->core()->render();
}


//-------------------------[ enableNormalization ]-------------------------\\

void GeometryGlyphsPlugin::enableNormalization(bool enable)
{
	if (this->builder == NULL)
		return;

	if (enable)
	{
		this->setNormalizationMethod(this->ui->normMethodCombo->currentIndex());
		return;
	}

	// If normalization has been disable, we set the method to "None"
	this->builder->setNormalizationMethod(vtkGeometryGlyphBuilder::NM_None);
	this->builder->Modified();
	this->core()->render();
}


//--------------------------[ enableSharpening ]---------------------------\\

void GeometryGlyphsPlugin::enableSharpening(bool enable)
{
	if (this->builder == NULL)
		return;

	this->builder->setEnableSharpening(enable);
	this->builder->Modified();
	this->core()->render();
}


//---------------------------[ enableSmoothing ]---------------------------\\

void GeometryGlyphsPlugin::enableSmoothing(bool enable)
{
	if (this->builder == NULL || this->smoothFilter == NULL)
		return;

	if (enable)
	{
		// If smoothing has been enabled, route the output of the glyph builder
		// through the smooth filter, and then to the mapper.

		this->smoothFilter->SetInput(this->builder->GetOutput());
		this->mapper->SetInput(this->smoothFilter->GetOutput());
	}
	else
	{
		// If smoothing has been disabled, directly connect the output of the
		// glyph builder to the input of the mapper.

		this->mapper->SetInput(this->builder->GetOutput());
	}

	this->core()->render();
}


//-------------------------[ updateSmoothOptions ]-------------------------\\

void GeometryGlyphsPlugin::updateSmoothOptions()
{
	if (this->smoothFilter == NULL)
		return;

	this->smoothFilter->SetNumberOfIterations(this->ui->smoothIterSpin->value());
	this->smoothFilter->SetRelaxationFactor(this->ui->smoothRelaxSpin->value());

	this->smoothFilter->Modified();
	this->core()->render();
}


//-----------------------------[ setGlyphType ]----------------------------\\

void GeometryGlyphsPlugin::setGlyphType(int index)
{
	if (this->builder == NULL)
		return;

	// Try to change the glyph type. If this fails (usually because the "3D Mesh"
	// type has been selected for a data set without triangulation), we switch back
	// to start-shaped glyphs. Error reporting in this case is handled by the
	// "setGlyphType" function.

	if (this->builder->setGlyphType((vtkGeometryGlyphBuilder::GeometryGlyphType) this->ui->glyphTypeCombo->currentIndex()) == false)
	{
		this->ui->glyphTypeCombo->setCurrentIndex(vtkGeometryGlyphBuilder::GGT_Star);
	}

	// Smoothing doesn't do anything for star-shaped glyphs, so if we're drawing
	// those glyphs, we might as well disable smoothing altogether.

	this->ui->smoothGroup->setEnabled((vtkGeometryGlyphBuilder::GeometryGlyphType) 
		this->ui->glyphTypeCombo->currentIndex() == vtkGeometryGlyphBuilder::GGT_Mesh);

	this->builder->Modified();
	this->core()->render();
}


//--------------------------[ changeColorMethod ]--------------------------\\

void GeometryGlyphsPlugin::changeColorMethod(int index)
{
	if (!(this->mapper) || !(this->builder))
		return;

	// If we're using scalar coloring or weighted direction coloring, set the scalar volume
	if (index == vtkGeometryGlyphBuilder::CM_WDirection || 
		index == vtkGeometryGlyphBuilder::CM_Scalar)
	{
		this->setScalarVolume(this->ui->colorScalarsCombo->currentIndex(), false);
	}

	// If we're using direction-based coloring, use the default coloring method
	if (index == vtkGeometryGlyphBuilder::CM_WDirection || 
		index == vtkGeometryGlyphBuilder::CM_Direction)
	{
		this->mapper->SetColorModeToDefault();
	}

	// Otherwise, map the scalars to colors through a LUT
	else
	{
		this->mapper->SetColorModeToMapScalars();
	}

	// Enable or disable controls
	this->ui->colorScalarsLabel->setEnabled(index == vtkGeometryGlyphBuilder::CM_WDirection || index == vtkGeometryGlyphBuilder::CM_Scalar);
	this->ui->colorScalarsCombo->setEnabled(index == vtkGeometryGlyphBuilder::CM_WDirection || index == vtkGeometryGlyphBuilder::CM_Scalar);
	this->ui->colorLUTLabel->setEnabled(index == vtkGeometryGlyphBuilder::CM_Radius || index == vtkGeometryGlyphBuilder::CM_Scalar);
	this->ui->colorLUTCombo->setEnabled(index == vtkGeometryGlyphBuilder::CM_Radius || index == vtkGeometryGlyphBuilder::CM_Scalar);

	// Set the coloring method
	this->builder->setColoringMethod((vtkGeometryGlyphBuilder::ColoringMethod) index);
	this->builder->Modified();
	this->core()->render();
}


//---------------------------[ setScalarVolume ]---------------------------\\

void GeometryGlyphsPlugin::setScalarVolume(int index, bool update)
{
	if (!(this->builder))
		return;

	// If we selected "None", clear the scalar volume pointer
	if (index == 0)
	{
		this->builder->setScalarVolume(NULL);

		if (update)
		{
			this->builder->Modified();
			this->core()->render();
		}

		return;
	}

	if ((index - 1) < 0 || (index - 1) >= this->scalarDataSets.size())
	{
		this->ui->colorScalarsCombo->setCurrentIndex(0);
		return;
	}

	// Get the scalar volume data set and image
	data::DataSet * ds = this->scalarDataSets[index - 1];
	vtkImageData * image = ds->getVtkImageData();

	// Update the scalar volume if it is empty
	if (image->GetActualMemorySize() == 0)
	{
		image->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	// Set the pointer, and render the scene
	this->builder->setScalarVolume(image);

	if (update)
	{
		this->builder->Modified();
		this->core()->render();
	}
}


//--------------------------------[ setLUT ]-------------------------------\\

void GeometryGlyphsPlugin::setLUT(int index)
{
	if (!(this->builder) || !(this->mapper))
		return;

	// Use the default LUT
	if (index == 0)
	{
		vtkLookupTable * defaultLUT = vtkLookupTable::New();

		// Default LUT has a range of 0-1
		defaultLUT->SetRange(0.0, 1.0);

		// Colors range from blue to red (rainbow)
		defaultLUT->SetHueRange(0.66667, 0.0);

		// Store the LUT
		this->mapper->SetLookupTable(defaultLUT);
		defaultLUT->Delete();

		// When using the default LUT, we normalize the scalar values to the
		// range 0-1, to make sure that they fit within the LUT range.

		this->builder->setNormalizeScalars(true);
		this->builder->Modified();

		return;
	}

	// Get the LUT from the selected data set
	data::DataSet * ds = this->lutDataSets[this->ui->colorLUTCombo->currentIndex() - 1];
	this->mapper->SetLookupTable(vtkScalarsToColors::SafeDownCast(ds->getVtkObject()));
	this->mapper->Modified();

	// When using a custom LUT, we do not normalize the scalars
	this->builder->setNormalizeScalars(false);
	this->builder->Modified();

	this->core()->render();
}


//-------------------------[ setTessellationOrder ]------------------------\\

void GeometryGlyphsPlugin::setTessellationOrder(int val)
{
	if (!(this->builder))
		return;

	this->builder->computeGeometry(val);
	this->builder->Modified();
	this->core()->render();
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libGeometryGlyphsPlugin, bmia::GeometryGlyphsPlugin)
