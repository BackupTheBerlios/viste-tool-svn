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
 * GpuGlyphsPlugin.cxx
 *
 * 2010-07-13	Tim Peeters
 * - First version
 *
 * 2010-12-16	Evert van Aart
 * - Added support for HARDI SH glyphs.
 * - Implemented "dataSetRemoved" and "dataSetChanged".
 *
 * 2011-01-10	Evert van Aart
 * - Added support for the Cylindrical Harmonics HARDI mapper.
 * - Added fused visualization.
 * - Automatically initialize fiber ODF when selected.
 * - Cleaned up the code, added comments.
 *
 * 2011-02-08	Evert van Aart
 * - Version 1.0.0.
 * - Added support for coloring DTI glyphs.
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.0.1.
 * - Made the two glyph actors non-pickable. This will prevent these actors from
 *   interfering with the Fiber Cutting plugin.
 *
 * 2011-03-29	Evert van Aart
 * - Version 1.0.2.
 * - Fused glyphs now correctly update when moving the planes. 
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.3.
 * - Correctly update when seed points are changed.
 *
 * 2011-06-20	Evert van Aart
 * - Version 1.1.0.
 * - Added LUTs for SH glyphs. Removed the range clamping and "SQRT" options for
 *   coloring, as both things can be achieved with the Transfer Functions.
 * - Measure values for SH glyphs coloring are now computed only once per seed
 *   point set and stored in an array, rather than at every render pass. This 
 *   should smoothen the camera movement for SH glyphs.
 * - Implemented "dataSetChanged" and "dataSetRemoved" for transfer functions.
 *
 * 2011-07-07	Evert van Aart
 * - Version 1.1.1.
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */


/** Includes */

#include "GpuGlyphsPlugin.h"
#include "vtkDTIGlyphMapperVA.h"
#include "vtkSHGlyphMapper.h"
#include "vtkCHGlyphMapper.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

GpuGlyphsPlugin::GpuGlyphsPlugin() : Plugin("GPU Glyphs")
{
	// Create a prop and a mapper for the DTI glyphs
	this->DTIMapper = vtkDTIGlyphMapperVA::New();
    this->DTIGlyphs = vtkVolume::New();
    this->DTIGlyphs->VisibilityOff();
    this->DTIGlyphs->SetMapper(this->DTIMapper);
	this->DTIGlyphs->SetPickable(0);

	// Create a prop for the HARDI glyphs. The mapper is created in "changeHARDIData".
	this->HARDIGlyphs = vtkVolume::New();
	this->HARDIGlyphs->VisibilityOff();
	this->HARDIGlyphs->SetPickable(0);

	// Add the two props to the assembly
	this->assembly = vtkPropAssembly::New();
	this->assembly->AddPart(this->DTIGlyphs);
	this->assembly->AddPart(this->HARDIGlyphs);

	// Set pointers to NULL
	this->HARDIMapper	= NULL;
	this->seedFilter	= NULL;
	this->dtiSeeds		= NULL;
	this->hardiSeeds	= NULL;

	// We use the SH HARDI mapper by default
	this->prevHARDIMapper = 0;

	// Create the GUI of the widget
    this->widget = new QWidget();
    this->ui = new Ui::GpuGlyphsForm();
    this->ui->setupUi(this->widget);

	// Add the short names of the available HARDI measures to the combo boxes
	for (int i = 0; i < HARDIMeasures::numberOfMeasures; ++i)
	{
		this->ui->coloringMeasureTypeCombo->addItem(QString(HARDIMeasures::GetShortName(i)));
		this->ui->fusedMeasureCombo->addItem(QString(HARDIMeasures::GetShortName(i)));
	}

	// Connect GUI elements to slot functions
	connect(this->ui->dtiCombo,					SIGNAL(currentIndexChanged(int)),	this, SLOT(changeDTIData()));
	connect(this->ui->hardiCombo,				SIGNAL(currentIndexChanged(int)),	this, SLOT(changeHARDIData()));
	connect(this->ui->seedsComboBox,			SIGNAL(currentIndexChanged(int)),	this, SLOT(seedsChanged()));
	connect(this->ui->scaleSpinBox,				SIGNAL(valueChanged(double)),		this, SLOT(setScale(double)));

	connect(this->ui->glyphsHARDIMapperCHRadio, SIGNAL(clicked()),					this, SLOT(changeHARDIData()));
	connect(this->ui->glyphsHARDIMapperSHRadio,	SIGNAL(clicked()),					this, SLOT(changeHARDIData()));

	connect(this->ui->dtiColorLUTCombo,			SIGNAL(currentIndexChanged(int)),	this, SLOT(changeDTIData()));
	connect(this->ui->dtiColorWeightCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeDTIData()));
	connect(this->ui->dtiColorLUTRadio,			SIGNAL(clicked()),					this, SLOT(changeDTIData()));
	connect(this->ui->dtiColorLightnessRadio,	SIGNAL(clicked()),					this, SLOT(changeDTIData()));
	connect(this->ui->dtiColorSaturationRadio,	SIGNAL(clicked()),					this, SLOT(changeDTIData()));
	connect(this->ui->dtiColorRGBRadio,			SIGNAL(clicked()),					this, SLOT(changeDTIData()));

	connect(this->ui->glyphTypeDTIRadio,		SIGNAL(clicked()),					this, SLOT(showDTIOnly()));
	connect(this->ui->glyphTypeHARDIRadio,		SIGNAL(clicked()),					this, SLOT(showHARDIOnly()));
	connect(this->ui->glyphTypeFusedRadio,		SIGNAL(clicked()),					this, SLOT(showFused()));

	connect(this->ui->dtiColorLUTRadio,			SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->dtiColorLightnessRadio,	SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->dtiColorSaturationRadio,	SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->dtiColorRGBRadio,			SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->glyphTypeDTIRadio,		SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->glyphTypeHARDIRadio,		SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->glyphTypeFusedRadio,		SIGNAL(clicked()),					this, SLOT(enableControls()));
	connect(this->ui->coloringCombo,			SIGNAL(currentIndexChanged(int)),	this, SLOT(enableControls()));

	connect(this->ui->localScalingCheck,		SIGNAL(clicked()),					this, SLOT(changeSHSettings()));
	connect(this->ui->minMaxNormCheck,			SIGNAL(clicked()),					this, SLOT(changeSHSettings()));
	connect(this->ui->fODFCheck,				SIGNAL(clicked()),					this, SLOT(changeSHSettings()));
	connect(this->ui->stepSizeSpin,				SIGNAL(valueChanged(double)),		this, SLOT(changeSHSettings()));
	connect(this->ui->refineSpin,				SIGNAL(valueChanged(int)),			this, SLOT(changeSHSettings()));
	connect(this->ui->coloringRadiusTSpin,		SIGNAL(valueChanged(int)),			this, SLOT(changeSHSettings()));

	connect(this->ui->coloringCombo,			SIGNAL(currentIndexChanged(int)),	this, SLOT(changeSHColoringMethod(int)));
	connect(this->ui->coloringMeasureTypeCombo, SIGNAL(currentIndexChanged(int)),	this, SLOT(clearHARDIScalars()));
	connect(this->ui->coloringSHLUTCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(clearHARDIScalars()));

	connect(this->ui->coloringMeasureTypeCombo, SIGNAL(currentIndexChanged(int)),	this, SLOT(changeSHSettings()));
	connect(this->ui->coloringSHLUTCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeSHSettings()));

	connect(this->ui->fusedThresholdSpin,		SIGNAL(valueChanged(double)),		this, SLOT(showFused()));
	connect(this->ui->fusedMeasureCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(showFused()));
	connect(this->ui->fusedInvertCheck,			SIGNAL(clicked()),					this, SLOT(showFused()));

}


//---------------------------------[ init ]--------------------------------\\

void GpuGlyphsPlugin::init()
{
	// Set the scale of the glyphs
    this->setScale(this->ui->scaleSpinBox->value());
}


//------------------------------[ Destructor ]-----------------------------\\

GpuGlyphsPlugin::~GpuGlyphsPlugin()
{
	// Delete objects created and maintained in this plugin
	if (this->DTIGlyphs)
		this->DTIGlyphs->Delete();

	if (this->HARDIGlyphs)
		this->HARDIGlyphs->Delete();

	if (this->DTIMapper)
		this->DTIMapper->Delete();

	if (this->HARDIMapper)
		this->HARDIMapper->Delete();

	if (this->dtiSeeds)
		this->dtiSeeds->Delete();

	if (this->hardiSeeds)
		this->hardiSeeds->Delete();

    // Clear the data set lists
	this->dtiImageDataSets.clear();
	this->hardiImageDataSets.clear();
	this->seedDataSets.clear();
	this->scalarDataSets.clear();
	this->lutDataSets.clear();

	// Unload the GUI
    delete this->widget; 

	// Delete the assembly
	this->assembly->Delete();
}


//------------------------------[ getVtkProp ]-----------------------------\\

vtkProp * GpuGlyphsPlugin::getVtkProp()
{
	// Return the assembly, which contains props for both DTI and HARDI glyphs
	return this->assembly;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * GpuGlyphsPlugin::getGUI()
{
	// Return the GUI widget
    return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void GpuGlyphsPlugin::dataSetAdded(data::DataSet * ds)
{
	// Assert the data set pointer (should never be NULL)
    Q_ASSERT(ds);

	// Get the kind of the data set
    QString kind = ds->getKind();

	// Currently supported image data sets are "eigen" (DTI eigensystem data,
	// used for visualizing DTI glyphs), "spherical harmonics" (used for
	// HARDI SH glyphs), "scalar volume" (used for coloring glyphs), and
	// "transfer function" (used for coloring glyphs).

    if (kind == "eigen") 
	{
		this->dtiImageAdded(ds);
	}

	else if (kind == "spherical harmonics")
	{
		this->hardiImageAdded(ds);
	}

    else if (kind == "seed points") 
	{
		this->seedsAdded(ds);
	}

	else if (kind == "scalar volume")
	{
		this->scalarsAdded(ds);
	}

	else if (kind == "transfer function")
	{
		this->lutAdded(ds);
	}
}


//------------------------------[ seedsAdded ]-----------------------------\\

void GpuGlyphsPlugin::seedsAdded(data::DataSet * ds)
{
    // Try to get the VTK object from the data set
    vtkObject * obj = ds->getVtkObject();

	// If no object has been set, we cannot use this data set
    if (!obj) 
	{
		return;
	}

	// Cast the object to a point set, which is the native format for seed points
    vtkPointSet * pointSet = vtkPointSet::SafeDownCast(obj);

	// Check if the casting succeeded
    if (!pointSet) 
	{
		return;
	}

    // Add the new data set to the list of currently available data sets
    this->seedDataSets.append(ds);

    // Add the new data set to the list of data sets in the GUI
    this->ui->seedsComboBox->addItem(ds->getName());
}


//----------------------------[ dtiImageAdded ]----------------------------\\

void GpuGlyphsPlugin::dtiImageAdded(data::DataSet * ds)
{
	// Try to get the image data from the data set
	vtkImageData * img = ds->getVtkImageData();

	// Check if the image has been set
	if (!img)
	{
		return;
	}

	// Check if the point data has been set
	if (!(img->GetPointData()))
	{
		return;
	}

	// Add the new data set to the list of currently available data sets
	this->dtiImageDataSets.append(ds);

	// Add the new data set to the list of data sets in the GUI
	this->ui->dtiCombo->addItem(ds->getName());

	this->enableControls();
}


//---------------------------[ hardiImageAdded ]---------------------------\\

void GpuGlyphsPlugin::hardiImageAdded(data::DataSet * ds)
{
	// Try to get the image data from the data set
	vtkImageData * img = ds->getVtkImageData();

	// Check if the image has been set
	if (!img)
	{
		return;
	}

	// Check if the point data has been set
	if (!(img->GetPointData()))
	{
		return;
	}

	// Add the new data set to the list of currently available data sets
	this->hardiImageDataSets.append(ds);

	// Add the new data set to the list of data sets in the GUI
	this->ui->hardiCombo->addItem(ds->getName());
	this->enableControls();
}


//-----------------------------[ scalarsAdded ]----------------------------\\

void GpuGlyphsPlugin::scalarsAdded(data::DataSet * ds)
{
	// For comments, see previous function
	vtkImageData * img = ds->getVtkImageData();

	if (!img)
		return;

	if (!(img->GetPointData()))
	{
		return;
	}

	this->scalarDataSets.append(ds);
	this->ui->dtiColorWeightCombo->addItem(ds->getName());
	this->enableControls();
}


//-------------------------------[ lutAdded ]------------------------------\\

void GpuGlyphsPlugin::lutAdded(data::DataSet * ds)
{
	// Try to downcast the LUT
	if (!(vtkColorTransferFunction::SafeDownCast(ds->getVtkObject())))
		return;

	this->lutDataSets.append(ds);
	this->ui->dtiColorLUTCombo->addItem(ds->getName());
	this->ui->coloringSHLUTCombo->addItem(ds->getName());
	this->enableControls();
}


//----------------------------[ dataSetChanged ]---------------------------\\

void GpuGlyphsPlugin::dataSetChanged(data::DataSet * ds)
{
	// DTI Images
	if (ds->getKind() == "eigen")
	{
		// Check if the data sets has been added to the plugin
		if (!(this->dtiImageDataSets.contains(ds)))
			return;

		// Check if the data set contains image data, remove it otherwise
		if (!(ds->getVtkImageData()))
			this->dataSetRemoved(ds);

		// Change the name of the data set in the GUI
		int dsId = this->dtiImageDataSets.indexOf(ds);
		this->ui->dtiCombo->setItemText(dsId, ds->getName());
	}
	// HARDI Images
	else if (ds->getKind() == "spherical harmonics")
	{
		if (!(this->hardiImageDataSets.contains(ds)))
			return;

		if (!(ds->getVtkImageData()))
			this->dataSetRemoved(ds);

		int dsId = this->hardiImageDataSets.indexOf(ds);
		this->ui->hardiCombo->setItemText(dsId, ds->getName());
	}
	// Seed points
	else if (ds->getKind() == "seed points")
	{
		if (!(this->seedDataSets.contains(ds)))
			return;

		if (!(ds->getVtkObject()))
			this->dataSetRemoved(ds);

		int dsId = this->seedDataSets.indexOf(ds);
		this->ui->seedsComboBox->setItemText(dsId, ds->getName());

		if (this->seedFilter)
			this->seedFilter->forceExecute();

		this->seedsChanged();

		this->core()->render();
	}
	// Scalar volumes
	else if (ds->getKind() == "scalar volume")
	{
		if (!(this->scalarDataSets.contains(ds)))
			return;

		if (!(ds->getVtkImageData()))
			this->dataSetRemoved(ds);

		int dsId = this->scalarDataSets.indexOf(ds);
		this->ui->dtiColorWeightCombo->setItemText(dsId, ds->getName());

		this->core()->render();
	}

	// Transfer Functions
	else if (ds->getKind() == "transfer function")
	{
		if (!(this->lutDataSets.contains(ds)))
			return;

		int dsId = this->lutDataSets.indexOf(ds);
		this->ui->dtiColorLUTCombo->setItemText(dsId, ds->getName());
		this->ui->coloringSHLUTCombo->setItemText(dsId, ds->getName());
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void GpuGlyphsPlugin::dataSetRemoved(data::DataSet * ds)
{
	// DTI Images
	if (ds->getKind() == "eigen")
	{
		// Check if the data sets has been added to the plugin
		if (!(this->dtiImageDataSets.contains(ds)))
			return;

		// Remove data set from the list and from the GUI
		int dsId = this->dtiImageDataSets.indexOf(ds);
		this->ui->dtiCombo->removeItem(dsId);
		this->dtiImageDataSets.removeAt(dsId);

		// Update the output and the GUI
		this->changeDTIData();
		this->enableControls();
	}
	// HARDI Images
	else if (ds->getKind() == "spherical harmonics")
	{
		if (!(this->hardiImageDataSets.contains(ds)))
			return;

		int dsId = this->hardiImageDataSets.indexOf(ds);
		this->ui->hardiCombo->removeItem(dsId);
		this->hardiImageDataSets.removeAt(dsId);
		this->changeHARDIData();
		this->enableControls();
	}
	// Seed points
	else if (ds->getKind() == "seed points")
	{
		if (!(this->seedDataSets.contains(ds)))
			return;

		int dsId = this->seedDataSets.indexOf(ds);
		this->ui->seedsComboBox->removeItem(dsId);
		this->seedDataSets.removeAt(dsId);
		this->changeDTIData();
		this->changeHARDIData();
		this->enableControls();
	}
	
	// Scalar volumes
	else if (ds->getKind() == "scalar volume")
	{
		// Check if the data sets has been added to the plugin
		if (!(this->scalarDataSets.contains(ds)))
			return;

		// Remove data set from the list and from the GUI
		int dsId = this->scalarDataSets.indexOf(ds);
		this->ui->dtiColorWeightCombo->removeItem(dsId);
		this->scalarDataSets.removeAt(dsId);

		// Update the output and the GUI
		this->changeDTIData();
		this->enableControls();
	}

	// Transfer Functions
	else if (ds->getKind() == "transfer function")
	{
		if (!(this->lutDataSets.contains(ds)))
			return;

		int dsId = this->lutDataSets.indexOf(ds);
		this->ui->dtiColorLUTCombo->removeItem(dsId);
		this->ui->coloringSHLUTCombo->removeItem(dsId);
		this->lutDataSets.removeAt(dsId);
	}
}


//----------------------------[ changeDTIData ]----------------------------\\

void GpuGlyphsPlugin::changeDTIData(vtkPointSet * seeds)
{
	// Nothing happens if we're not displaying DTI glyphs
	if (!(this->ui->glyphTypeDTIRadio->isChecked()) && !(this->ui->glyphTypeFusedRadio->isChecked()))
	{
		return;
	}

	// Get the indices of the selected data sets
	int imageIndex = this->ui->dtiCombo->currentIndex();
	int seedIndex  = this->ui->seedsComboBox->currentIndex();

	// Check if the indices are within the correct range
	if ( imageIndex < 0 || imageIndex >= this->dtiImageDataSets.size() ||
		  seedIndex < 0 ||  seedIndex >= this->seedDataSets.size()  )
	{
		// If not, turn off visibility of the data sets
		this->DTIGlyphs->VisibilityOff();
        
		return;
	}

	// Get the data sets from the lists of data set pointers
	data::DataSet * dsImage = this->dtiImageDataSets.at(imageIndex);
    data::DataSet * dsSeeds = this->seedDataSets.at(seedIndex);

	// Seed points used for the glyphs
	vtkPointSet * points;

	// If defined, use the "seeds" set (used for fused visualization)
	if (seeds)
	{
		points = seeds;
	}
	// Otherwise, use the seed point set defined by the combo box
	else
	{
		// Get the seed points from the data set
		vtkObject * obj = dsSeeds->getVtkObject();
		points = vtkPointSet::SafeDownCast(obj);
	}

	// Get the image data from the data set
    vtkImageData * image = dsImage->getVtkImageData();

	// Create the DTI mapper
	if (!this->DTIMapper)
	{
		this->DTIMapper = vtkDTIGlyphMapperVA::New();
		this->DTIGlyphs->SetMapper(this->DTIMapper);
	}

	vtkObject * attObject = NULL;

	// If available, add the transformation matrix to the mapper
	if (dsImage->getAttributes()->getAttribute("transformation matrix", attObject))
	{
		// Note: Setting the user matrix for the actor ("DTIGlyphs") does not actually
		// transform the glyphs, because of the way the ray caster is set up. To transform
		// the glyphs, we also need to add this matrix to our custom mapper, "DTIMapper".
		// However, adding the matrix to the actor is still necessary, because otherwise,
		// resetting the camera will not take this transformation into account, leading to
		// an erroneous focal point when translating the image.

		this->DTIGlyphs->SetUserMatrix(vtkMatrix4x4::SafeDownCast(attObject));
		this->DTIMapper->setTransformationMatrix(vtkMatrix4x4::SafeDownCast(attObject));
	}
	// If not, set the transformation matrix to NULL
	else
	{
		this->DTIGlyphs->SetUserMatrix(NULL);
		this->DTIMapper->setTransformationMatrix(NULL);
	}

	// Set the coloring method for the DTI glyphs
	vtkDTIGlyphMapperVA::ColoringMethod coloringMethod = vtkDTIGlyphMapperVA::CM_RGB;

	if (this->ui->dtiColorSaturationRadio->isChecked())		coloringMethod = vtkDTIGlyphMapperVA::CM_WRGBA;
	if (this->ui->dtiColorLightnessRadio->isChecked())		coloringMethod = vtkDTIGlyphMapperVA::CM_WRGBB;
	if (this->ui->dtiColorLUTRadio->isChecked())			coloringMethod = vtkDTIGlyphMapperVA::CM_LUT;

	this->DTIMapper->setColoringMethod(coloringMethod);

	// Set the scalar volume data set
	if (!(this->scalarDataSets.isEmpty()))
	{
		data::DataSet * ds = this->scalarDataSets.at(this->ui->dtiColorWeightCombo->currentIndex());

		if(ds)
		{
			vtkImageData * image = ds->getVtkImageData();

			if (image)
			{
				// We need to update the image, since AI images are computed on request
				if (image->GetActualMemorySize() == 0)
				{
					image->Update();
					this->core()->data()->dataSetChanged(ds);
				}

				this->DTIMapper->setAIImage(ds->getVtkImageData());
			}
		}
	}
	else
	{
		this->DTIMapper->setAIImage(NULL);
	}

	// Set the transfer function data set
	if (!(this->lutDataSets.isEmpty()))
	{
		data::DataSet * ds = this->lutDataSets.at(this->ui->dtiColorLUTCombo->currentIndex());

		if(ds)
		{
			vtkScalarsToColors * lut = vtkScalarsToColors::SafeDownCast(ds->getVtkObject());
			this->DTIMapper->setLUT(lut);
		}
	}
	else
	{
		this->DTIMapper->setLUT(NULL);
	}

	// Setup the mapper
    this->DTIMapper->SetInput(image);
    this->DTIMapper->SetSeedPoints(points);

	// Render the DTI glyphs
    this->DTIGlyphs->VisibilityOn();
    this->core()->render();
}


//---------------------------[ changeHARDIData ]---------------------------\\

void GpuGlyphsPlugin::changeHARDIData(vtkPointSet * seeds)
{
	// Nothing happens if we're not displaying HARDI glyphs
	if (!(this->ui->glyphTypeHARDIRadio->isChecked()) && !(this->ui->glyphTypeFusedRadio->isChecked()))
	{
		return;
	}

	// Get the indices of the selected data sets
	int imageIndex = this->ui->hardiCombo->currentIndex();
	int seedIndex  = this->ui->seedsComboBox->currentIndex();

	// Check if the indices are within the correct range
	if ( imageIndex < 0 || imageIndex >= this->hardiImageDataSets.size() ||
		  seedIndex < 0 ||  seedIndex >= this->seedDataSets.size()       )
	{
		// If not, turn off visibility of the data sets
		this->HARDIGlyphs->VisibilityOff();
        
		return;
	}

	// Get the data sets from the lists of data set pointers
	data::DataSet * dsImage = this->hardiImageDataSets.at(imageIndex);
    data::DataSet * dsSeeds = this->seedDataSets.at(seedIndex);

	// Seed points used for the glyphs
	vtkPointSet * points;

	// If defined, use the "seeds" set (used for fused visualization)
	if (seeds)
	{
		points = seeds;
	}
	// Otherwise, use the seed point set defined by the combo box
	else
	{
		// Get the seed points from the data set
		vtkObject * obj = dsSeeds->getVtkObject();
		points = vtkPointSet::SafeDownCast(obj);
	}

	// Get the image data from the data set
    vtkImageData * image = dsImage->getVtkImageData();

	// Determine the mapper type (SH or CH)
	int newHARDIMapper = (this->ui->glyphsHARDIMapperSHRadio->isChecked()) ? 0 : 1;

	// If the mapper type has changed, delete the current mapper
	if (this->prevHARDIMapper != newHARDIMapper)
	{
		this->HARDIMapper->Delete();
		this->HARDIMapper = NULL;
		this->prevHARDIMapper = newHARDIMapper;
	}

	// Create a new HARDI glyphs mapper
	if (!this->HARDIMapper)
	{
		// Spherical Harmonics mapper
		if (newHARDIMapper == 0)
		{
			this->HARDIMapper = vtkSHGlyphMapper::New();
		}
		// Cylindrical Harmonics mapper
		else
		{
			this->HARDIMapper = vtkCHGlyphMapper::New();
		}

		// Store the new mapper
		this->HARDIGlyphs->SetMapper(this->HARDIMapper);
	}
	
	vtkObject * attObject = NULL;

	// If available, add the transformation matrix to the mapper
	if (dsImage->getAttributes()->getAttribute("transformation matrix", attObject))
	{
		vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(attObject);

		this->HARDIGlyphs->SetUserMatrix(m);
		this->HARDIMapper->setTransformationMatrix(m);

		if (m->Element[0][1] != 0.0 || m->Element[0][2] != 0.0 ||
			m->Element[1][0] != 0.0 || m->Element[1][2] != 0.0 || 
			m->Element[2][0] != 0.0 || m->Element[2][1] != 0.0)
		{
			QString err = "";
			err += "The selected data set has a transformation matrix which includes rotation, ";
			err += "but the SH glyph mappers currently do not support rotation. ";
			err += "The orientation of the glyphs will be wrong. Please consider changing the transformation matrix.";
			QMessageBox::warning(this->getGUI(), "Transformation Matrix", err);
		}
	}
	// If not, set the transformation matrix to NULL
	else
	{
		this->HARDIGlyphs->SetUserMatrix(NULL);
		this->HARDIMapper->setTransformationMatrix(NULL);
	}

	// Set the transfer function data set
	if (!(this->lutDataSets.isEmpty()))
	{
		data::DataSet * ds = this->lutDataSets.at(this->ui->coloringSHLUTCombo->currentIndex());

		if(ds)
		{
			vtkScalarsToColors * lut = vtkScalarsToColors::SafeDownCast(ds->getVtkObject());
			this->HARDIMapper->setLUT(lut);
		}
	}
	else
	{
		this->HARDIMapper->setLUT(NULL);
	}

	// Setup the mapper
	this->HARDIMapper->SetInput(image);
	this->HARDIMapper->SetSeedPoints(points);
	this->HARDIGlyphs->VisibilityOn();
	this->changeSHSettings();
}


//-------------------------------[ setScale ]------------------------------\\

void GpuGlyphsPlugin::setScale(double scale)
{
	// Set scale for DTI glyphs
	if (this->DTIMapper)
	{
		this->DTIMapper->SetGlyphScaling(scale);
	}

	// Set scale for HARDI glyphs
	if (this->HARDIMapper)
	{
		this->HARDIMapper->SetGlyphScaling(scale);
	}

    this->core()->render();
}


//---------------------------[ changeSHSettings ]--------------------------\\

void GpuGlyphsPlugin::changeSHSettings()
{
	// Do nothing if no HARDI mapper exists
	if (this->HARDIMapper == NULL)
		return;

	// Copy settings from the GUI to the mapper
	this->HARDIMapper->SetLocalScaling(this->ui->localScalingCheck->isChecked());
	this->HARDIMapper->SetMinMaxNormalize(this->ui->minMaxNormCheck->isChecked());
	this->HARDIMapper->SetFiberODF(this->ui->fODFCheck->isChecked());
	this->HARDIMapper->SetStepSize(this->ui->stepSizeSpin->value());
	this->HARDIMapper->SetNumRefineSteps(this->ui->refineSpin->value());
	this->HARDIMapper->SetColoring(this->ui->coloringCombo->currentIndex());
	this->HARDIMapper->setColoringMeasure(this->ui->coloringMeasureTypeCombo->currentIndex());
	this->HARDIMapper->SetRadiusThreshold(this->ui->coloringRadiusTSpin->value() / 100.0f);

	// If fiber ODF sharpening is enabled, initialize it now
	if (this->ui->fODFCheck->isChecked())
	{
		// Get the index of the selected eigensystem image
		int imageIndex = this->ui->dtiCombo->currentIndex();

		// Check if the index is within the correct range
		if (imageIndex < 0 || imageIndex >= this->dtiImageDataSets.size())
		{
			return;
		}

		// Get the data set from the lists of data set pointers
		data::DataSet * dsImage = this->dtiImageDataSets.at(imageIndex);

		// Use the eigensystem image to initialize fiber ODF
		this->HARDIMapper->initODF(dsImage->getVtkImageData());
	}

	// Update the mapper and render the glyphs
	this->HARDIMapper->Modified();
	this->HARDIMapper->Update();
	this->core()->render();
}


//--------------------------[ clearHARDIScalars ]--------------------------\\

void GpuGlyphsPlugin::clearHARDIScalars()
{
	if (this->HARDIMapper)
		this->HARDIMapper->clearScalars();
}


//------------------------[ changeSHColoringMethod ]-----------------------\\

void GpuGlyphsPlugin::changeSHColoringMethod(int index)
{
	// If we're trying to switch to LUTs, but there are no LUTs available...
	if (index == (int) vtkSHGlyphMapper::SHCM_Measure && this->lutDataSets.isEmpty())
	{
		// ...show a message and switch back to RGB coloring
		this->core()->out()->showMessage("Cannot change coloring method: No available LUTs.", "GPU Glyphs");
		this->ui->coloringCombo->setCurrentIndex(0);
		return;
	}

	// Otherwise, just update the settings for SH
	this->changeSHSettings();
}

//----------------------------[ enableControls ]---------------------------\\

void GpuGlyphsPlugin::enableControls()
{
	// Disable DTI/HARDI if no data sets are available
	bool enableDTI =   (this->ui->dtiCombo->count()   != 0);
	bool enableHARDI = (this->ui->hardiCombo->count() != 0);

	// Glyphs type radio buttons
	this->ui->glyphsTypeBox->setEnabled(enableDTI || enableHARDI);
	this->ui->glyphTypeDTIRadio->setEnabled(enableDTI);
	this->ui->glyphTypeHARDIRadio->setEnabled(enableHARDI);
	this->ui->glyphTypeFusedRadio->setEnabled(enableDTI && enableHARDI);

	// Force-select DTI or HARDI glyphs if the other type is
	// no longer available (because of deleted data sets).

	if (enableDTI && !enableHARDI)
	{
		this->ui->glyphTypeDTIRadio->setChecked(true);
		this->changeDTIData();
	}
	else if (!enableDTI && enableHARDI)
	{
		this->ui->glyphTypeHARDIRadio->setChecked(true);
		this->changeHARDIData();
	}

	// Disable/enable fused visualization options
	bool fusedChecked = this->ui->glyphTypeFusedRadio->isChecked();
	this->ui->fusedBox->setEnabled(fusedChecked);

	// Disable/enable DTI glyph options
	bool dtiChecked =	   this->ui->glyphTypeDTIRadio->isChecked()
						|| this->ui->glyphTypeFusedRadio->isChecked();

	this->ui->dtiColorGroup->setEnabled(dtiChecked);
	this->ui->dtiColorLightnessRadio->setEnabled(dtiChecked && this->ui->dtiColorWeightCombo->count() > 0);
	this->ui->dtiColorSaturationRadio->setEnabled(dtiChecked && this->ui->dtiColorWeightCombo->count() > 0);
	this->ui->dtiColorLUTRadio->setEnabled(dtiChecked && this->ui->dtiColorLUTCombo->count() > 0);

	bool enableDTILUTCombo = dtiChecked && this->ui->dtiColorLUTRadio->isChecked() && this->ui->dtiColorLUTRadio->isEnabled();
	this->ui->dtiColorLUTLabel->setEnabled(enableDTILUTCombo);
	this->ui->dtiColorLUTCombo->setEnabled(enableDTILUTCombo);
	
	bool enableDTIWeightCombo = dtiChecked && 
								((this->ui->dtiColorLightnessRadio->isChecked()  && this->ui->dtiColorLightnessRadio->isEnabled() ) ||
								 (this->ui->dtiColorSaturationRadio->isChecked() && this->ui->dtiColorSaturationRadio->isEnabled()) );

	this->ui->dtiColorWeightLabel->setEnabled(enableDTIWeightCombo);
	this->ui->dtiColorWeightCombo->setEnabled(enableDTIWeightCombo);

	// Disable/enable HARDI glyph options
	bool hardiChecked =    this->ui->glyphTypeHARDIRadio->isChecked()
						|| this->ui->glyphTypeFusedRadio->isChecked();

	this->ui->glyphsHARDIMapperBox->setEnabled(hardiChecked);
	this->ui->shapeBox->setEnabled(hardiChecked);
	this->ui->rayCastingBox->setEnabled(hardiChecked);
	this->ui->coloringBox->setEnabled(hardiChecked);

	// Disable/enable coloring measure options
	bool coloringMeasureEnable = hardiChecked && this->ui->coloringCombo->currentIndex() == (int) vtkSHGlyphMapper::SHCM_Measure;
	this->ui->coloringMeasureTypeLabel->setEnabled(coloringMeasureEnable);
	this->ui->coloringMeasureTypeCombo->setEnabled(coloringMeasureEnable);
	this->ui->coloringSHLUTLabel->setEnabled(coloringMeasureEnable);
	this->ui->coloringSHLUTCombo->setEnabled(coloringMeasureEnable);

	// Disable/enable radius coloring options
	bool coloringRadiusEnable = hardiChecked && this->ui->coloringCombo->currentIndex() == (int) vtkSHGlyphMapper::SHCM_Radius;
	this->ui->coloringRadiusTLabel->setEnabled(coloringRadiusEnable);
	this->ui->coloringRadiusTSlide->setEnabled(coloringRadiusEnable);
	this->ui->coloringRadiusTSpin->setEnabled(coloringRadiusEnable);

}


//-----------------------------[ showDTIOnly ]-----------------------------\\

void GpuGlyphsPlugin::showDTIOnly()
{
	// Hide HARDI glyphs, update DTI glyphs
	this->HARDIGlyphs->VisibilityOff();
	this->changeDTIData();
}


//----------------------------[ showHARDIOnly ]----------------------------\\

void GpuGlyphsPlugin::showHARDIOnly()
{
	// Hide DTI glyphs, update HARDI glyphs
	this->DTIGlyphs->VisibilityOff();
	this->changeHARDIData();
}


//------------------------------[ showFused ]------------------------------\\

void GpuGlyphsPlugin::showFused()
{
	// Delete existing seed point sets
	if (this->dtiSeeds)
	{
		this->dtiSeeds->Delete();
		this->dtiSeeds = NULL;
	}

	if (this->hardiSeeds)
	{
		this->hardiSeeds->Delete();
		this->hardiSeeds = NULL;
	}

	// Create new point sets for DTI and HARDI
	this->dtiSeeds = vtkPointSet::SafeDownCast(vtkUnstructuredGrid::New());
	this->hardiSeeds = vtkPointSet::SafeDownCast(vtkUnstructuredGrid::New());

	// Create a threshold filter
	if (!(this->seedFilter))
	{
		this->seedFilter = vtkThresholdFilter::New();
	}

	// Setup the threshold filter
	this->seedFilter->setOutputs(this->dtiSeeds, this->hardiSeeds);
	this->seedFilter->setThreshold(this->ui->fusedThresholdSpin->value());
	this->seedFilter->setInvert(this->ui->fusedInvertCheck->isChecked());

	// Get the indices of the selected data sets
	int imageIndex = this->ui->hardiCombo->currentIndex();
	int seedIndex  = this->ui->seedsComboBox->currentIndex();

	// Check if the indices are within the correct range
	if ( imageIndex < 0 || imageIndex >= this->hardiImageDataSets.size() ||
		 seedIndex < 0 ||  seedIndex >= this->seedDataSets.size()       )
	{
		// If not, turn off visibility of the data sets
		this->DTIGlyphs->VisibilityOff();
		this->HARDIGlyphs->VisibilityOff();

		return;
	}

	// Get the data sets from the lists of data set pointers
	data::DataSet * dsImage = this->hardiImageDataSets.at(imageIndex);
	data::DataSet * dsSeeds = this->seedDataSets.at(seedIndex);
	
	// Set inputs of the threshold filter
	this->seedFilter->SetInput(vtkPointSet::SafeDownCast(dsSeeds->getVtkObject()));
	this->seedFilter->setSHImage(dsImage->getVtkImageData());
	this->seedFilter->setMeasure((HARDIMeasures::HARDIMeasureType) this->ui->fusedMeasureCombo->currentIndex());

	// Force the filter to update
	this->seedFilter->forceExecute();

	// Use the output point sets as seed points for the glyphs
	this->changeDTIData(this->seedFilter->GetOutput(0));
	this->changeHARDIData(this->seedFilter->GetOutput(1));
}


//----------------------------[ seedsChanged ]-----------------------------\\

void GpuGlyphsPlugin::seedsChanged()
{
	// Show DTI, HARDI, or fused glyphs, depending on GUI settings
	if (this->ui->glyphTypeDTIRadio->isChecked())
	{
		this->showDTIOnly();
	}
	else if (this->ui->glyphTypeHARDIRadio->isChecked())
	{
		this->showHARDIOnly();
	}
	else
	{
		this->showFused();
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libGpuGlyphsPlugin, bmia::GpuGlyphsPlugin)
