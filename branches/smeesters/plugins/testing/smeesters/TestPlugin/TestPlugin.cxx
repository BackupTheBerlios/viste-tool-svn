#include "TestPlugin.h"

#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

// init values
#define DEFAULT_SMOOTHING 1.0
#define DEFAULT_ALPHA 1.0
#define DEFAULT_SPECULAR 0.00

namespace bmia
{

///
///      INITIALIZATION
///

//------------------------[ Plugin constructor ]-----------------------\\

TestPlugin::TestPlugin() : plugin::AdvancedPlugin("TestPlugin")
{
    this->widget = NULL;
    this->form   = NULL;
}

//------------------------[ Plugin destructor ]-----------------------\\

TestPlugin::~TestPlugin()
{
    delete this->widget;
    delete this->form;
}

//------------------------[ Initialization ]-----------------------\\

void TestPlugin::init()
{
    this->widget = new QWidget();
    this->form = new Ui::TestPluginForm();
    this->form->setupUi(this->widget);

    // Link events in the GUI to function calls
    this->connectAll();
    this->assembly = vtkPropAssembly::New();

    this->current_modelInfo = NULL;
    this->overlay_modelInfo = NULL;

    // Update the combo box
    this->comboBoxDataChanged();

    // setup picker
    this->setupClippingPlanesPicker();

    // Get the canvas
	vtkMedicalCanvas * canvas = this->fullCore()->canvas();

	QString sliceActorNames[3];
    sliceActorNames[0] = "X Plane";
    sliceActorNames[1] = "Y Plane";
    sliceActorNames[2] = "Z Plane";

	// Loop through all three axes
	for(int axis = 0; axis < 3; ++axis)
	{
        vtkSmartPointer<vtkSphereSource> diskSource =
            vtkSmartPointer<vtkSphereSource>::New();
        diskSource->SetRadius(4);

        vtkSmartPointer<vtkPolyDataMapper> diskMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
        diskMapper->SetInputConnection(diskSource->GetOutputPort());

        vtkActor* diskActor =
            vtkActor::New();
        diskActor->SetMapper(diskMapper);
        diskActor->SetPosition(180,216,0);
        diskActor->GetProperty()->SetColor(1,0,0);

        canvas->GetSubCanvas2D(axis)->GetRenderer()->AddActor(diskActor);

        this->pointer2DList.append(diskActor);
	}

	// Add measured points
	for(int i = 0; i < 2; i++)
	{
        MeasuredPoint* point = new MeasuredPoint;

        point->set = false;

        vtkSmartPointer<vtkSphereSource> diskSource =
            vtkSmartPointer<vtkSphereSource>::New();
        diskSource->SetRadius(2);

        vtkSmartPointer<vtkPolyDataMapper> diskMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
        diskMapper->SetInputConnection(diskSource->GetOutputPort());

        vtkActor* diskActor =
            vtkActor::New();
        diskActor->SetMapper(diskMapper);

        if(i == 0)
            diskActor->GetProperty()->SetColor(1,0,0);
        else
            diskActor->GetProperty()->SetColor(0,1,0);


        point->sphere = diskActor;

        this->measuredPointList.append(point);
	}

	this->measuredLine = NULL;

}

//------------[ Setup pointer interactor for clipping planes ]----------------\\

void TestPlugin::setupClippingPlanesPicker()
{
    // Create a new styleTrackballPP
    this->styleTrackballPP = vtkInteractorStyleTrackballPositionPicker::New();

    // Set up the options for styleTrackballPP
    this->styleTrackballPP->SetRenderProcess(this->fullCore()->canvas()->GetRenderer3D());
    this->styleTrackballPP->SetParentClass(this);

    // Set this styleTrackballPP to subcanvas
    vtkSubCanvas * subcanvas = this->fullCore()->canvas()->GetSubCanvas3D();
    subcanvas->SetInteractorStyle(this->styleTrackballPP);

    // Create a cellPicker and set up its options
    vtkCellPicker * cellPicker = vtkCellPicker::New();

    // Set this cellPicker to subcanvas
    subcanvas->GetInteractor()->SetPicker(cellPicker);
    cellPicker->Delete();
}

///
///      DATA I/O
///

//------------------------[ Dataset added ]-----------------------\\

void TestPlugin::dataSetAdded(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Load scalar volume
    if(kind == "scalar volume")
    {
        // Keep track of the datasets used by this plugin
        this->datasets.append(d);

        // Add to UI combobox for selection of data
        this->form->comboBoxDataset->addItem(d->getName());

        // Add to UI combobox for selection of overlay
        this->form->comboBoxOverlay->addItem(d->getName());

        // Create model info
        createModelInfo(d);
    }

    // Transfer Functions (LUTs)
	else if (kind == "transfer function")
	{
	    // Keep track of transfer function datasets
	    this->tf_datasets.append(d);

        // Add to UI combobox for selection of overlay plane LUT
		this->form->comboBoxOverlayLUT->addItem(d->getName());

		// Add to UI combobox for selection of base layer plane LUT
		this->form->comboBoxBaseLayerLUT->addItem(d->getName());

        // Create the lookup table from the transfer function
        createLookupTable(d);
	}
}

//------------------------[ Dataset changed ]-----------------------\\

void TestPlugin::dataSetChanged(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Scalar volume
    if(kind == "scalar volume")
    {
        // Get index
        int dsIndex = this->datasets.indexOf(d);

        // Change names in comboboxes
        this->form->comboBoxDataset->setItemText(dsIndex,d->getName());
        this->form->comboBoxOverlay->setItemText(dsIndex,d->getName());
    }

    // Transfer Functions (LUTs)
    else if (kind == "transfer function")
	{
	    // Check if the data set has been added to this plugin
        if (!(this->tf_datasets.contains(d)))
            return;

        // Change the name of the data set in the GUI
        int index = this->tf_datasets.indexOf(d);

        // Create new transfer function LUTs
        createLookupTable(d,index);

        // Update the LUTs
        comboBoxOverlayLUTChanged();
        comboBoxBaseLayerLUTChanged();
	}
}

//------------------------[ Dataset removed ]-----------------------\\

void TestPlugin::dataSetRemoved(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Scalar volume
    if(kind == "scalar volume")
    {
        // Check if the data set has been added to this plugin
        if (!(this->datasets.contains(d)))
            return;

        // Get index
        int dsIndex = this->datasets.indexOf(d);

        // Remove from UI combobox for selection of data
        this->form->comboBoxDataset->removeItem(dsIndex);

        // Remove from UI combobox for selection of overlay
        this->form->comboBoxOverlay->removeItem(dsIndex);

        // Remove model info
        this->modelInfoList.removeAt(dsIndex);
    }

    // Transfer Functions (LUTs)
    else if (kind == "transfer function")
	{

	}
}



///
///      PROCESSING
///

//------------------------[ Create model info ]-----------------------\\

void TestPlugin::createModelInfo(data::DataSet * d)
{
    // Create model info struct for new dataset
    ModelInfo* modelInfo = new ModelInfo;
    modelInfo->ds = d;
    modelInfo->alpha = DEFAULT_ALPHA;
    modelInfo->prop = NULL; // null pointer
    modelInfo->polydata = NULL;
    modelInfo->colorString = "0xFFFFFF";
    modelInfo->color[0] = modelInfo->color[1] = modelInfo->color[2] = 1.0;  // corresponding to white: #FFFFFF
    modelInfo->renderStyle = 0;
    modelInfo->reduction = 0.0;
    modelInfo->visible = true;
    modelInfo->specular = DEFAULT_SPECULAR;
    modelInfo->selectLargestComponent = true;

    modelInfo->extractPolyFunc = NULL;

    d->getVtkImageData()->GetScalarRange(modelInfo->scalarRange);
    modelInfo->maximumThreshold = ceil(modelInfo->scalarRange[1]*0.75); // automatic upper range
    modelInfo->minimumThreshold = floor(modelInfo->scalarRange[1]*0.25);    // automatic lower range
    if(modelInfo->minimumThreshold == 0)    // zero lower threshold makes model invisible
        modelInfo->minimumThreshold += 0.01;
    modelInfo->smoothing = ceil(modelInfo->scalarRange[1]) == 1.0 ? 0.0 : DEFAULT_SMOOTHING;    // turn smoothing off for binary image

    d->getVtkImageData()->GetBounds(modelInfo->bounds);
    modelInfo->clippingValues[0] = modelInfo->bounds[0];
    modelInfo->clippingValues[1] = modelInfo->bounds[2];
    modelInfo->clippingValues[2] = modelInfo->bounds[4];

    modelInfo->clippingCube = vtkBox::New();
    modelInfo->clippingCube->SetBounds(modelInfo->bounds);
    modelInfo->invertClipping = false;

    modelInfo->planesActivated[0] = modelInfo->planesActivated[1] = modelInfo->planesActivated[2] = false;
    modelInfo->flipped[0] = modelInfo->flipped[1] = modelInfo->flipped[2] = false;
    modelInfo->dsOverlay = NULL;
    modelInfo->overlayIndex = 0;

    modelInfo->bDsPolyAdded = false;

    modelInfo->alignPlanesToPick = true;

    // create lookup table for grey values
    vtkLookupTable* defaultLUT = vtkLookupTable::New();
    defaultLUT->SetValueRange(0.0, 1.0);
    defaultLUT->SetSaturationRange(0.0, 0.0);
    defaultLUT->SetAlphaRange(0.0, 1.0);
    double range[2];
    d->getVtkImageData()->GetScalarRange(range);
    defaultLUT->SetTableRange(range);
    defaultLUT->Build();

    modelInfo->defaultLUT = defaultLUT;

    // Create the three orthogonal planes for 2D image merging
    double normals[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int i = 0; i < 3; i++)
    {
        vtkPlane* slicePlane = vtkPlane::New();
        slicePlane->SetNormal(normals[i]);
        slicePlane->SetOrigin(0,0,0);

        vtkImageResliceMapper* imageMapper = vtkImageResliceMapper::New();
        imageMapper->SetInput(d->getVtkImageData());
        imageMapper->SetSlicePlane(slicePlane);
        imageMapper->SetSeparateWindowLevelOperation(0);
        imageMapper->SliceFacesCameraOff();
        imageMapper->SliceAtFocalPointOff();

        vtkImageSlice *image = vtkImageSlice::New();
        image->SetMapper(imageMapper);
        image->Update();
        imageMapper->Delete();
        image->GetProperty()->SetLookupTable(defaultLUT);
        image->GetProperty()->SetUseLookupTableScalarRange(1);

        // Use the identity matrix by default
        vtkMatrix4x4 * id = vtkMatrix4x4::New();
        id->Identity();
        image->PokeMatrix(id);

        vtkObject* tfm;
        if (d->getAttributes()->getAttribute("transformation matrix", tfm ))
        {
            vtkMatrix4x4* transformationMatrix = vtkMatrix4x4::SafeDownCast(tfm);
            if (transformationMatrix == 0)
            {
                this->core()->out()->logMessage("not a valid transformation matrix");
                return;
            }
            image->PokeMatrix(transformationMatrix);
        }
        else
        {
            double* center;
            center = d->getVtkImageData()->GetCenter();
            image->SetOrigin(center[0],center[1],center[2]);
        }

        // add model to list
        modelInfo->orthogonalPlanesModels.append(image);

        // add plane to list
        modelInfo->orthogonalPlanes.append(slicePlane);

        // by default invisible
        image->VisibilityOff();

        double bounds[6];
        image->GetBounds(bounds);
        modelInfo->global_bounds[i*2] = bounds[i*2];
        modelInfo->global_bounds[i*2+1] = bounds[i*2+1];

        this->assembly->AddPart(image);
        image->Delete();
    }

    // Add to local model list
    this->modelInfoList.append(modelInfo);

//    // Add to dataset manager so that it shows up in the PolyData list
//    // Set the default name for the new ROI
//	QString newDataName = QString("TestPlugin: ").append(d->getName());
//
//	// Create an empty polydata object
//	vtkPolyData * polyData = vtkPolyData::New();
//
//	// Create a data set with the default name and the empty data
//	data::DataSet * ds_poly = new data::DataSet(newDataName, "regionOfInterest", polyData);
//	polyData->Delete();
//
//	modelInfo->ds_poly = ds_poly;
}

//------------------------[ Process transfer function ]-----------------------\\

void TestPlugin::createLookupTable(data::DataSet * d, int index)
{
	 // get color transfer function
    vtkColorTransferFunction* ctf = vtkColorTransferFunction::SafeDownCast(d->getVtkObject());
    int numberOfColorNodes = ctf->GetSize();
    std::cout << "Number of nodes: " << numberOfColorNodes << std::endl;

    // color range
    double colorRange[2];
    ctf->GetRange(colorRange);
	int numberOfValues = ceil(colorRange[1]);

	// get piecewise function attribute
    vtkObject* cpf;
	vtkPiecewiseFunction* pf;
	int numberOfOpacityNodes;
    if(d->getAttributes()->getAttribute("piecewise function", cpf ))
	{
		// get opacity transfer function
		pf = vtkPiecewiseFunction::SafeDownCast(cpf);
		numberOfOpacityNodes = pf->GetSize();
		pf->Print(std::cout);
		std::cout << "Number of nodes: " << numberOfOpacityNodes << std::endl;
	}

	// no opacity transfer function available, create one (no opacity: 1.0 to 1.0)
	else
	{
		pf = vtkPiecewiseFunction::New();
		pf->AdjustRange(colorRange);

		double point1[4] = {0.0, 1.0, 0.0, 0.5};
		pf->SetNodeValue(0,point1);

		double point2[4] = {colorRange[1], 1.0, 0.0, 0.5};
		pf->SetNodeValue(0,point2);
	}

	// opacity range
    double opacityRange[2];
    pf->GetRange(opacityRange);

    // create new lookup table object
    vtkLookupTable* convLUT = vtkLookupTable::New();
    convLUT->SetScaleToLinear();
    convLUT->SetValueRange(0.0, 92.0);
    convLUT->SetNumberOfTableValues(numberOfValues);
    convLUT->SetTableRange(0,numberOfValues);
    convLUT->Build();

    // store the data from the color and opacity transfer functions in qlists
    QList<double> colorIndices;
    QList<double> colorR;
    QList<double> colorG;
    QList<double> colorB;
    QList<double> alpha;

    for(int j = 0; j<numberOfColorNodes; j++)
    {
        double colorVal[6];
        ctf->GetNodeValue(j,colorVal);

        double opacityVal[4];
        pf->GetNodeValue(j,opacityVal);

        colorIndices.append(colorVal[0]);
        colorR.append(colorVal[1]);
        colorG.append(colorVal[2]);
        colorB.append(colorVal[3]);

        alpha.append(opacityVal[1]);
    }

    // fill the lookup table with data. interpolate points between nodes of the transfer function.
    for(int j = 0; j<numberOfValues; j++)
    {
        // linear interpolation
        int ind = -1;
        for(int k = 0; k<colorIndices.length(); k++)
        {
            if(j < colorIndices.at(k))
            {
                ind = std::max(0,k-1);
                break;
            }
        }

        if(ind == -1)
            continue;

        double cl_ind = colorIndices.at(ind);
        double cl_r = colorR.at(ind);
        double cl_g = colorG.at(ind);
        double cl_b = colorB.at(ind);
        double al_v = alpha.at(ind);

        ind = ind + 1;
        double cu_ind = colorIndices.at(ind);
        double cu_r = colorR.at(ind);
        double cu_g = colorG.at(ind);
        double cu_b = colorB.at(ind);
        double au_v = alpha.at(ind);

        double ind_frac = j-cl_ind;
        ind_frac /= (double)(cu_ind - cl_ind);

        convLUT->SetTableValue(j,
                               cl_r*(1-ind_frac) + cu_r*(ind_frac),
                               cl_g*(1-ind_frac) + cu_g*(ind_frac),
                               cl_b*(1-ind_frac) + cu_b*(ind_frac),
                               al_v*(1-ind_frac) + au_v*(ind_frac));
    }

    if(index == -1)
        lookUpTables.append(convLUT);
    else
        lookUpTables.replace(index,convLUT);
}

//------------------------[ Create or update the mesh ]-----------------------\\

void TestPlugin::updateRenderingModels()
{
    // recalculate model
    if(bModelDirty)
    {
        // remove old actor (make conditional)
        if(current_modelInfo->prop != NULL)
        {
            this->assembly->RemovePart(current_modelInfo->prop);
            current_modelInfo->prop = NULL;
        }

        data::DataSet* ds = this->current_modelInfo->ds;

        // smooth dataset
        VTK_CREATE(vtkImageGaussianSmooth, smoother);
        smoother->SetInput(ds->getVtkImageData());
        smoother->SetStandardDeviation(current_modelInfo->smoothing,current_modelInfo->smoothing,current_modelInfo->smoothing);

        // marching cubes
        VTK_CREATE(vtkMarchingCubes, mcubes);
        mcubes->SetInputConnection(smoother->GetOutputPort());
        mcubes->SetNumberOfContours(1);
        mcubes->SetComputeScalars(0);
        mcubes->GenerateValues(1,current_modelInfo->minimumThreshold,current_modelInfo->maximumThreshold);
        //mcubes->GetOutput()->GetBounds(current_modelInfo->prop_bounds);

        // select largest component
        VTK_CREATE(vtkPolyDataConnectivityFilter, connectivity);
        if(current_modelInfo->selectLargestComponent)
        {
            connectivity->SetInputConnection(mcubes->GetOutputPort());
            connectivity->SetExtractionModeToLargestRegion();
        }

        // depth sort for correct alpha display
//        VTK_CREATE(vtkDepthSortPolyData, depthSort);
//        depthSort->SetInputConnection(connectivity->GetOutputPort());
//        depthSort->SetDirectionToBackToFront();
//        depthSort->SetVector(1,1,1);
//        depthSort->SetCamera(this->fullCore()->canvas()->GetRenderer3D()->GetActiveCamera());
//        depthSort->SortScalarsOn();
//        depthSort->Update();

        // decimate
        VTK_CREATE(vtkDecimatePro, deci);
        if(current_modelInfo->reduction > 0.1)
        {
            if(current_modelInfo->selectLargestComponent)
                deci->SetInputConnection(connectivity->GetOutputPort());
            else
                deci->SetInputConnection(mcubes->GetOutputPort());
            deci->SetTargetReduction(current_modelInfo->reduction);
            deci->PreserveTopologyOn();
            deci->AccumulateErrorOn();
        }

        vtkExtractPolyDataGeometry* clipper = vtkExtractPolyDataGeometry::New();
        clipper->SetImplicitFunction(current_modelInfo->clippingCube);
        if(current_modelInfo->reduction > 0.1)
            clipper->SetInputConnection(deci->GetOutputPort());
        else
		{
			if(current_modelInfo->selectLargestComponent)
                clipper->SetInputConnection(connectivity->GetOutputPort());
            else
                clipper->SetInputConnection(mcubes->GetOutputPort());
		}
        clipper->ExtractInsideOn();
        current_modelInfo->extractPolyFunc = clipper;

        // Create a mapper and actor
        VTK_CREATE(vtkPolyDataMapper,polyMapper);
        polyMapper->SetInputConnection(clipper->GetOutputPort());

        VTK_CREATE(vtkLODActor,actor);
        actor->SetMapper(polyMapper);
        actor->GetProperty()->SetColor(current_modelInfo->color);
        if(current_modelInfo->alpha < 1.0)
        {
            actor->GetProperty()->SetOpacity(current_modelInfo->alpha);
        }
        actor->GetProperty()->SetBackfaceCulling(1);
        actor->GetProperty()->SetInterpolationToPhong();
        actor->GetProperty()->SetSpecular(current_modelInfo->specular);
        actor->GetProperty()->SetSpecularPower(2.0);

        actor->SetOrientation(0,0,0);
        actor->SetPosition(0,0,0);
        actor->SetScale(1,1,1);
        actor->SetOrigin(0,0,0);

        // Use the identity matrix by default
        vtkMatrix4x4 * id = vtkMatrix4x4::New();
        id->Identity();
        actor->PokeMatrix(id);

        vtkObject* tfm;
        if (ds->getAttributes()->getAttribute("transformation matrix", tfm ))
        {
            vtkMatrix4x4* transformationMatrix = vtkMatrix4x4::SafeDownCast(tfm);
            if (transformationMatrix == 0)
            {
                this->core()->out()->logMessage("not a valid transformation matrix");
                return;
            }
            actor->PokeMatrix(transformationMatrix);

            //current_modelInfo->ds_poly->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(transformationMatrix));
        }
        else
        {
            double* center;
            center = actor->GetCenter();
            actor->SetOrigin(center[0],center[1],center[2]);
        }

        id->Delete();
        tfm->Delete();

        current_modelInfo->polydata = connectivity->GetOutput();
        current_modelInfo->prop = actor;
        this->assembly->AddPart(actor);

		for(int i =0; i<3; i++)
        {
            this->updateClippingPlaneSlider(i,current_modelInfo->clippingValues[i]);
        }
        // Reset the camera of the 3D volume
        //this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();

        // Reset clipping rnage
       // this->fullCore()->canvas()->GetRenderer3D()->ResetCameraClippingRange();

//        // Update the polydata of the DataSet or the manager
//        current_modelInfo->ds_poly->updateData(clipper->GetOutput());
//
//        // Add the model to the Data manager if not already done
//        if(!(current_modelInfo->bDsPolyAdded))
//        {
//            this->core()->data()->addDataSet(current_modelInfo->ds_poly);
//            std::cout << "ADD DATA!";
//            current_modelInfo->bDsPolyAdded = true;
//        }
//
//        // else Signal the core that the data set has changed
//        else
//        {
//            this->core()->data()->dataSetChanged(current_modelInfo->ds_poly);
//        }



    }

    // adjust existing model
    else
    {
        current_modelInfo->prop->GetProperty()->SetColor(current_modelInfo->color);
        current_modelInfo->prop->GetProperty()->SetOpacity(current_modelInfo->alpha);
        current_modelInfo->prop->GetProperty()->SetSpecular(current_modelInfo->specular);
    }

    bModelDirty = false;
    this->core()->render();

    //vtkDebugLeaks::PrintCurrentLeaks();
}

///
///     CALLBACKS
///

//------------[ Callback of clipping planes position from picker ]----------------\\

void TestPlugin::setClippingPlanesPosition(double* pos)
{
    if(this->current_modelInfo != NULL)
    {
        double local_pos[3];
        double frac_pos[3];

        for(int i = 0; i<3; i++)
        {

            double n_pos = (double)(pos[i]-current_modelInfo->global_bounds[i*2])/(abs(current_modelInfo->global_bounds[i*2] - current_modelInfo->global_bounds[i*2+1]));
            frac_pos[i] = n_pos;
            n_pos *= current_modelInfo->bounds[i*2+1];
            local_pos[i] = n_pos;
            this->clickedPoint[i] = pos[i];

            std::cout << i << " " << pos[i] << " --- " << n_pos << std::endl;
        }

        //this->clickedPoint = local_pos;

        QString sliceActorNames[3];
        sliceActorNames[0] = "X Plane";
        sliceActorNames[1] = "Y Plane";
        sliceActorNames[2] = "Z Plane";

        double sliceActorRanges[3];
        for(int axis = 0; axis < 3; ++axis)
        {
            std::cout << "AXIS: " << axis << std::endl;
            vtkImageSliceActor * sliceActor = static_cast<vtkImageSliceActor*>(this->core()->data()->getDataSet(sliceActorNames[axis], "sliceActor")->getVtkObject());
            //sliceActor->Print(std::cout);
            //sliceActor->GetUserMatrix()->Print(std::cout);

            vtkActor* pointer = this->pointer2DList.at(axis);
            pointer->SetUserMatrix(sliceActor->GetUserMatrix());
        }

        //vtkMedicalCanvas * canvas = this->fullCore()->canvas();

        //vtkImageSliceActor * sliceActor = this->fullCore()->data()->getDataSet()

        for(int i = 0; i<3; i++)
        {
            vtkActor* pointer = this->pointer2DList.at(i);

            switch(i)
            {
                case 0: // sagittal
                    pointer->SetPosition(frac_pos[0]*18,frac_pos[1]*217,frac_pos[2]*181);
                    break;

                case 1: // coronal
                    pointer->SetPosition(frac_pos[0]*181,300,frac_pos[2]*181);
                    break;

                case 2: // axial
                    pointer->SetPosition(frac_pos[0]*181,frac_pos[1]*217,-90);
                    break;
            }
        }

        if(!current_modelInfo->alignPlanesToPick)
            return;

        vtkPoints* points = vtkPoints::New();
        int radius = 5;
        int numPoints = 0;
        for(int i = -1; i<=1; i++)
        {
            for(int j = -1; j<=1; j++)
            {
                for(int k = -1; k<=1; k++)
                {
                    double neighbour_pos[3];
                    neighbour_pos[0] = local_pos[0] + i*radius;
                    neighbour_pos[1] = local_pos[1] + j*radius;
                    neighbour_pos[2] = local_pos[2] + k*radius;
                    points->InsertNextPoint(neighbour_pos);

                    //std::cout << "npoint: " << neighbour_pos[0] << " " << neighbour_pos[1] << " " << neighbour_pos[2] << std::endl;

                    numPoints++;
                }
            }
        }

        vtkPolyData* pointsPolyData = vtkPolyData::New();
        pointsPolyData->SetPoints(points);

        vtkSelectEnclosedPoints* selectEnclosedPoints = vtkSelectEnclosedPoints::New();
        selectEnclosedPoints->SetInput(pointsPolyData);
        selectEnclosedPoints->SetSurface(current_modelInfo->polydata);
        selectEnclosedPoints->Update();

        bool isInside = false;
        for(int i = 0; i<numPoints; i++)
        {
            if(selectEnclosedPoints->IsInside(i))
                isInside = true;
        }
        std::cout << "inside: " << isInside << std::endl;


        if(isInside)
        {
            for(int i = 0; i<3; i++)
            {
                if(!current_modelInfo->planesActivated[i])
                    continue;

                this->updateClippingPlaneSlider(i,(int)local_pos[i],false);

                switch(i)
                {
                    case 0:
                        this->form->horizontalSliderX->setValue(local_pos[i]);
                        break;

                    case 1:
                        this->form->horizontalSliderY->setValue(local_pos[i]);
                        break;

                    case 2:
                        this->form->horizontalSliderZ->setValue(local_pos[i]);
                        break;
                }
            }
        }






        //this->core()->render();

        /*std::cout << "xrange: " << current_modelInfo->global_bounds[0] << " -- " << current_modelInfo->global_bounds[1] << std::endl;
        std::cout << "yrange: " << current_modelInfo->global_bounds[2] << " -- " << current_modelInfo->global_bounds[3] << std::endl;
        std::cout << "zrange: " << current_modelInfo->global_bounds[4] << " -- " << current_modelInfo->global_bounds[5] << std::endl;*/
    }


}

///
///     UPDATE FROM USER INPUT
///

//-----------[ Update clipping plane slider effects on mesh ]-----------------------\\

void TestPlugin::updateClippingPlaneSlider(int direction, int value, bool render)
{
    // return if no model is chosen
    if(current_modelInfo == NULL)
        return;

    //std::cout << "Update clipping plane slider: " << direction << " value: " << value << std::endl;

    // adjust bounds of clipping cube used to cut away the mesh
    double cubeBounds[6];
    current_modelInfo->clippingCube->GetBounds(cubeBounds);

    // flipped, reverse bounds
    if(current_modelInfo->flipped[direction])
    {
        cubeBounds[direction*2] = current_modelInfo->bounds[direction*2]; // minimum bound for direction
        cubeBounds[direction*2+1] = value;
    }

    // not flipped
    else
    {
        cubeBounds[direction*2] = value;
        cubeBounds[direction*2+1] = current_modelInfo->bounds[direction*2+1]; // maximum bound for direction
    }

    current_modelInfo->clippingCube->SetBounds(cubeBounds);
    current_modelInfo->clippingValues[direction] = value;

    // compute origin of orthogonal clipping planes in order to move it
    double origin[3];
    origin[direction] = (value - this->current_modelInfo->bounds[direction*2+1]/2.0);

    // temporary origin correction
    if(direction == 1) { origin[direction] -= 18; }
    if(direction == 2) { origin[direction] += 18; }

    //if(direction == 0) { origin[direction] -= 6; }
    //if(direction == 1) { origin[direction] += 10; }
    //if(direction == 2) { origin[direction] -= 4; }


    vtkPlane* plane = this->current_modelInfo->orthogonalPlanes.at(direction);
    plane->SetOrigin(origin);

    // overlay plane visibility and position
    if(this->overlay_modelInfo != NULL)
    {
        vtkPlane* plane_overlay = this->overlay_modelInfo->orthogonalPlanes.at(direction);
        if(direction == 0)
            origin[direction] -= -(current_modelInfo->flipped[direction]*2 - 1)*0.1;
        else if(direction == 1)
            origin[direction] -= -(current_modelInfo->flipped[direction]*2 - 1)*0.1;
        else if(direction == 2)
            origin[direction] += -(current_modelInfo->flipped[direction]*2 - 1)*0.1;
        plane_overlay->SetOrigin(origin);
    }

    if(render)
        this->core()->render();
}

//-----------[ Update clipping plane enabled checkbox action ]-----------------------\\

void TestPlugin::updateClippingPlaneEnabled(int direction, bool checked)
{
    current_modelInfo->planesActivated[direction] = checked;

    // orthogonal plane visibility and position
    vtkImageSlice* planeModel = this->current_modelInfo->orthogonalPlanesModels.at(direction);
    planeModel->SetVisibility(checked);

    // overlay plane visibility and position
    if(this->overlay_modelInfo != NULL)
    {
        vtkImageSlice* planeModel_overlay = this->overlay_modelInfo->orthogonalPlanesModels.at(direction);
        planeModel_overlay->SetVisibility(checked);
    }

    this->updateClippingPlaneSlider(direction,current_modelInfo->clippingValues[direction]);
}


///
///      GUI CONTROLS
///

//------------------------[ Connect Qt elements ]-----------------------\\

void TestPlugin::connectAll()
{
    connect(this->form->comboBoxDataset,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxDataChanged()));
    connect(this->form->checkBoxVisible,SIGNAL(toggled(bool)),this,SLOT(checkBoxVisibleChanged(bool)));
    connect(this->form->buttonUpdate,SIGNAL(clicked()),this,SLOT(buttonUpdateClicked()));
    connect(this->form->inputMaximumThreshold,SIGNAL(valueChanged(double)),this,SLOT(inputMaximumThresholdChanged(double)));
    connect(this->form->inputMinimumThreshold,SIGNAL(valueChanged(double)),this,SLOT(inputMinimumThresholdChanged(double)));
    connect(this->form->inputSmoothing,SIGNAL(valueChanged(double)),this,SLOT(inputSmoothingChanged(double)));
    connect(this->form->inputColor,SIGNAL(textChanged(QString)),this,SLOT(inputColorChanged(QString)));
    connect(this->form->inputAlpha,SIGNAL(valueChanged(double)),this,SLOT(inputAlphaChanged(double)));
    connect(this->form->inputReduction,SIGNAL(valueChanged(double)),this,SLOT(inputReductionChanged(double)));
    connect(this->form->inputSpecular,SIGNAL(valueChanged(double)),this,SLOT(inputSpecularChanged(double)));
    connect(this->form->horizontalSliderX,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderXChanged(int)));
    connect(this->form->horizontalSliderY,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderYChanged(int)));
    connect(this->form->horizontalSliderZ,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderZChanged(int)));
    connect(this->form->checkBoxX,SIGNAL(toggled(bool)),this,SLOT(checkBoxXChanged(bool)));
    connect(this->form->checkBoxY,SIGNAL(toggled(bool)),this,SLOT(checkBoxYChanged(bool)));
    connect(this->form->checkBoxZ,SIGNAL(toggled(bool)),this,SLOT(checkBoxZChanged(bool)));
    connect(this->form->comboBoxStyle,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxStyleChanged()));
    connect(this->form->comboBoxOverlay,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxOverlayChanged()));
    connect(this->form->comboBoxOverlayLUT,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxOverlayLUTChanged()));
    connect(this->form->comboBoxBaseLayerLUT,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxBaseLayerLUTChanged()));
    connect(this->form->checkBoxFlipX,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipXChanged(bool)));
    connect(this->form->checkBoxFlipY,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipYChanged(bool)));
    connect(this->form->checkBoxFlipZ,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipZChanged(bool)));
    connect(this->form->buttonSaveMesh,SIGNAL(clicked()),this,SLOT(buttonSaveMeshClicked()));
    connect(this->form->spinX,SIGNAL(valueChanged(int)),this,SLOT(spinXChanged(int)));
    connect(this->form->spinY,SIGNAL(valueChanged(int)),this,SLOT(spinYChanged(int)));
    connect(this->form->spinZ,SIGNAL(valueChanged(int)),this,SLOT(spinZChanged(int)));
    connect(this->form->checkBoxInvertClipping,SIGNAL(toggled(bool)),this,SLOT(checkBoxInvertClippingChanged(bool)));
    connect(this->form->checkBoxLargestComponent,SIGNAL(toggled(bool)),this,SLOT(checkBoxLargestComponentChanged(bool)));
    connect(this->form->checkBoxAlignPlanesToPick,SIGNAL(toggled(bool)),this,SLOT(checkBoxAlignPlanesToPickChanged(bool)));

    connect(this->form->buttonSetPointA,SIGNAL(clicked()),this,SLOT(buttonSetPointAClicked()));
    connect(this->form->buttonSetPointB,SIGNAL(clicked()),this,SLOT(buttonSetPointBClicked()));

    connect(this->form->buttonSetLineColor,SIGNAL(clicked()),this,SLOT(buttonSetLineColorClicked()));
    connect(this->form->buttonSaveMeasurement,SIGNAL(clicked()),this,SLOT(buttonSaveMeasurementClicked()));


}

//------------------------[ Disconnect Qt elements ]-----------------------\\

void TestPlugin::disconnectAll()
{
    disconnect(this->form->comboBoxDataset,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxDataChanged()));
    disconnect(this->form->checkBoxVisible,SIGNAL(toggled(bool)),this,SLOT(checkBoxVisibleChanged(bool)));
    disconnect(this->form->buttonUpdate,SIGNAL(clicked()),this,SLOT(buttonUpdateClicked()));
    disconnect(this->form->inputMaximumThreshold,SIGNAL(valueChanged(double)),this,SLOT(inputMaximumThresholdChanged(double)));
    disconnect(this->form->inputMinimumThreshold,SIGNAL(valueChanged(double)),this,SLOT(inputMinimumThresholdChanged(double)));
    disconnect(this->form->inputSmoothing,SIGNAL(valueChanged(double)),this,SLOT(inputSmoothingChanged(double)));
    disconnect(this->form->inputColor,SIGNAL(textChanged(QString)),this,SLOT(inputColorChanged(QString)));
    disconnect(this->form->inputAlpha,SIGNAL(valueChanged(double)),this,SLOT(inputAlphaChanged(double)));
    disconnect(this->form->inputReduction,SIGNAL(valueChanged(double)),this,SLOT(inputReductionChanged(double)));
    disconnect(this->form->inputSpecular,SIGNAL(valueChanged(double)),this,SLOT(inputSpecularChanged(double)));
    disconnect(this->form->horizontalSliderX,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderXChanged(int)));
    disconnect(this->form->horizontalSliderY,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderYChanged(int)));
    disconnect(this->form->horizontalSliderZ,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderZChanged(int)));
    disconnect(this->form->checkBoxX,SIGNAL(toggled(bool)),this,SLOT(checkBoxXChanged(bool)));
    disconnect(this->form->checkBoxY,SIGNAL(toggled(bool)),this,SLOT(checkBoxYChanged(bool)));
    disconnect(this->form->checkBoxZ,SIGNAL(toggled(bool)),this,SLOT(checkBoxZChanged(bool)));
    disconnect(this->form->comboBoxStyle,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxStyleChanged()));
    disconnect(this->form->comboBoxOverlay,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxOverlayChanged()));
    disconnect(this->form->comboBoxOverlayLUT,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxOverlayLUTChanged()));
    disconnect(this->form->comboBoxBaseLayerLUT,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxBaseLayerLUTChanged()));
    disconnect(this->form->checkBoxFlipX,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipXChanged(bool)));
    disconnect(this->form->checkBoxFlipY,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipYChanged(bool)));
    disconnect(this->form->checkBoxFlipZ,SIGNAL(toggled(bool)),this,SLOT(checkBoxFlipZChanged(bool)));
    disconnect(this->form->buttonSaveMesh,SIGNAL(clicked()),this,SLOT(buttonSaveMeshClicked()));
    disconnect(this->form->spinX,SIGNAL(valueChanged(int)),this,SLOT(spinXChanged(int)));
    disconnect(this->form->spinY,SIGNAL(valueChanged(int)),this,SLOT(spinYChanged(int)));
    disconnect(this->form->spinZ,SIGNAL(valueChanged(int)),this,SLOT(spinZChanged(int)));
}

//------------------------[ Select another dataset from combobox ]-----------------------\\

void TestPlugin::comboBoxDataChanged()
{
    // select model info matching the dataset
    int index = this->form->comboBoxDataset->currentIndex() - 1;

    // "None" value
    if(index < 0)
    {
        // disable gui
        this->form->checkBoxVisible->setEnabled(false);
        this->form->buttonUpdate->setEnabled(false);
        this->form->inputAlpha->setEnabled(false);
        this->form->inputColor->setEnabled(false);
        this->form->inputMaximumThreshold->setEnabled(false);
        this->form->inputMinimumThreshold->setEnabled(false);
        this->form->inputSmoothing->setEnabled(false);
        this->form->inputSpecular->setEnabled(false);
        this->form->checkBoxX->setEnabled(false);
        this->form->checkBoxY->setEnabled(false);
        this->form->checkBoxZ->setEnabled(false);
        this->form->inputReduction->setEnabled(false);
        this->form->comboBoxStyle->setEnabled(false);
        this->form->comboBoxOverlay->setEnabled(false);
        this->form->comboBoxOverlayLUT->setEnabled(false);
        this->form->horizontalSliderX->setEnabled(false);
        this->form->horizontalSliderY->setEnabled(false);
        this->form->horizontalSliderZ->setEnabled(false);
        this->form->checkBoxFlipX->setEnabled(false);
        this->form->checkBoxFlipY->setEnabled(false);
        this->form->checkBoxFlipZ->setEnabled(false);
        this->form->buttonApplySettingsToAll->setEnabled(false);
        this->form->comboBoxBaseLayer->setEnabled(false);
        this->form->comboBoxBaseLayerLUT->setEnabled(false);
        this->form->spinX->setEnabled(false);
        this->form->spinY->setEnabled(false);
        this->form->spinZ->setEnabled(false);
        this->form->buttonSaveMesh->setEnabled(false);
        this->form->checkBoxLargestComponent->setEnabled(false);
        this->form->checkBoxInvertClipping->setEnabled(false);
        this->form->checkBoxAlignPlanesToPick->setEnabled(false);
    }
    else
    {
        // enable gui
        this->form->checkBoxVisible->setEnabled(true);
        this->form->buttonUpdate->setEnabled(true);
        this->form->inputAlpha->setEnabled(true);
        this->form->inputColor->setEnabled(true);
        this->form->inputMaximumThreshold->setEnabled(true);
        this->form->inputMinimumThreshold->setEnabled(true);
        this->form->inputSmoothing->setEnabled(true);
        this->form->inputSpecular->setEnabled(true);
        this->form->checkBoxX->setEnabled(true);
        this->form->checkBoxY->setEnabled(true);
        this->form->checkBoxZ->setEnabled(true);
        this->form->inputReduction->setEnabled(true);
        this->form->comboBoxStyle->setEnabled(true);
        this->form->comboBoxOverlay->setEnabled(true);
        this->form->comboBoxOverlayLUT->setEnabled(true);
        this->form->horizontalSliderX->setEnabled(true);
        this->form->horizontalSliderY->setEnabled(true);
        this->form->horizontalSliderZ->setEnabled(true);
        this->form->checkBoxFlipX->setEnabled(true);
        this->form->checkBoxFlipY->setEnabled(true);
        this->form->checkBoxFlipZ->setEnabled(true);
        this->form->buttonApplySettingsToAll->setEnabled(true);
        this->form->comboBoxBaseLayerLUT->setEnabled(true);
        this->form->spinX->setEnabled(true);
        this->form->spinY->setEnabled(true);
        this->form->spinZ->setEnabled(true);
        this->form->buttonSaveMesh->setEnabled(true);
        this->form->checkBoxLargestComponent->setEnabled(true);
        this->form->checkBoxInvertClipping->setEnabled(true);
        this->form->checkBoxAlignPlanesToPick->setEnabled(true);

        // select model info struct
        this->current_modelInfo = this->modelInfoList.at(index);

        // update gui values
        this->form->horizontalSliderX->setMinimum(current_modelInfo->bounds[0]);
        this->form->horizontalSliderX->setMaximum(current_modelInfo->bounds[1]);
        this->form->horizontalSliderY->setMinimum(current_modelInfo->bounds[2]);
        this->form->horizontalSliderY->setMaximum(current_modelInfo->bounds[3]);
        this->form->horizontalSliderZ->setMinimum(current_modelInfo->bounds[4]);
        this->form->horizontalSliderZ->setMaximum(current_modelInfo->bounds[5]);
        this->form->horizontalSliderX->setValue(current_modelInfo->clippingValues[0]);
        this->form->horizontalSliderY->setValue(current_modelInfo->clippingValues[1]);
        this->form->horizontalSliderZ->setValue(current_modelInfo->clippingValues[2]);

        this->form->spinX->setMinimum(current_modelInfo->bounds[0]);
        this->form->spinX->setMaximum(current_modelInfo->bounds[1]);
        this->form->spinY->setMinimum(current_modelInfo->bounds[2]);
        this->form->spinY->setMaximum(current_modelInfo->bounds[3]);
        this->form->spinZ->setMinimum(current_modelInfo->bounds[4]);
        this->form->spinZ->setMaximum(current_modelInfo->bounds[5]);
        this->form->spinX->setValue(current_modelInfo->clippingValues[0]);
        this->form->spinY->setValue(current_modelInfo->clippingValues[1]);
        this->form->spinZ->setValue(current_modelInfo->clippingValues[2]);

        this->form->inputAlpha->setValue(current_modelInfo->alpha);
        this->form->inputSmoothing->setValue(current_modelInfo->smoothing);
        this->form->inputColor->setText(current_modelInfo->colorString);
        this->form->inputSpecular->setValue(current_modelInfo->specular);
        this->form->checkBoxVisible->setChecked(current_modelInfo->visible);
        this->form->inputReduction->setValue(current_modelInfo->reduction);
        this->form->comboBoxStyle->setCurrentIndex(current_modelInfo->renderStyle);
        this->form->checkBoxLargestComponent->setChecked(current_modelInfo->selectLargestComponent);
        this->form->checkBoxInvertClipping->setChecked(current_modelInfo->invertClipping);

        this->form->inputMinimumThreshold->setMinimum(current_modelInfo->scalarRange[0]);
        this->form->inputMaximumThreshold->setMaximum(current_modelInfo->scalarRange[1]);
        this->form->inputMaximumThreshold->setValue(current_modelInfo->maximumThreshold);
        this->form->inputMinimumThreshold->setValue(current_modelInfo->minimumThreshold);

        this->form->checkBoxX->setChecked(current_modelInfo->planesActivated[0]);
        this->form->checkBoxY->setChecked(current_modelInfo->planesActivated[1]);
        this->form->checkBoxZ->setChecked(current_modelInfo->planesActivated[2]);

        this->form->checkBoxFlipX->setChecked(current_modelInfo->flipped[0]);
        this->form->checkBoxFlipY->setChecked(current_modelInfo->flipped[1]);
        this->form->checkBoxFlipZ->setChecked(current_modelInfo->flipped[2]);

        // enable all items in overlay combobox
        int numberOfItems = this->form->comboBoxOverlay->count();
        for(int i = 1; i<numberOfItems; i++)
        {
            QModelIndex ind = this->form->comboBoxOverlay->model()->index(i, 0);
            QVariant v(1 | 32);
            this->form->comboBoxOverlay->model()->setData(ind, v, Qt::UserRole - 1);
        }

        // disable current dataset item in overlay combobox
        QModelIndex ind = this->form->comboBoxOverlay->model()->index(index + 1, 0); // index + 1 to skip "None"
        QVariant v(0);
        this->form->comboBoxOverlay->model()->setData(ind, v, Qt::UserRole - 1);

        // set overlay combobox index
        this->form->comboBoxOverlay->setCurrentIndex(this->current_modelInfo->overlayIndex);
        this->form->comboBoxOverlayLUT->setCurrentIndex(this->current_modelInfo->overlayLUTIndex);

        // set base layer combobox index
        this->form->comboBoxBaseLayer->setCurrentIndex(index);
        this->form->comboBoxBaseLayerLUT->setCurrentIndex(this->current_modelInfo->baseLayerLUTIndex);

        // update overlay
        //this->comboBoxOverlayChanged();

        // mark model update required
		bModelDirty = true;


    }
}

//---------------[ Select overlay dataset for clipping planes ]-----------------\\

void TestPlugin::comboBoxOverlayChanged()
{
    // select model info matching the dataset
    int index = this->form->comboBoxOverlay->currentIndex();

    this->current_modelInfo->overlayIndex = index;

    if((index - 1) >= 0) // Skip "None" value
    {
        this->overlay_modelInfo = this->modelInfoList.at(index - 1);

        this->updateClippingPlaneEnabled(0,current_modelInfo->planesActivated[0]);
        this->updateClippingPlaneEnabled(1,current_modelInfo->planesActivated[1]);
        this->updateClippingPlaneEnabled(2,current_modelInfo->planesActivated[2]);
    }
    else
    {
        this->overlay_modelInfo = NULL;
    }

    this->core()->render();
}

//---------------[ Select overlay LUT for clipping planes ]-----------------\\

void TestPlugin::comboBoxOverlayLUTChanged()
{
    // select model info matching the dataset
    int index = this->form->comboBoxOverlayLUT->currentIndex();

    if(this->overlay_modelInfo != NULL)
    {
        this->current_modelInfo->overlayLUTIndex = index;

        if(index > 0)
        {
            vtkLookupTable* ctf = this->lookUpTables.at(index - 1);
            for(int i = 0; i<3; i++)
            {
                this->overlay_modelInfo->orthogonalPlanesModels[i]->GetProperty()->SetLookupTable(ctf);
            }
        }

        // use default LUT
        else if(index == 0)
        {
            for(int i = 0; i<3; i++)
            {
                this->overlay_modelInfo->orthogonalPlanesModels[i]->GetProperty()->SetLookupTable(this->overlay_modelInfo->defaultLUT);
            }
        }
    }

    this->core()->render();
}

//---------------[ Select base layer LUT for clipping planes ]-----------------\\

void TestPlugin::comboBoxBaseLayerLUTChanged()
{
    // select model info matching the dataset
    int index = this->form->comboBoxBaseLayerLUT->currentIndex();

    if(this->current_modelInfo != NULL)
    {
        this->current_modelInfo->baseLayerLUTIndex = index;

		if(index > 0)
        {
            vtkLookupTable* ctf = this->lookUpTables.at(index - 1);
            for(int i = 0; i<3; i++)
            {
                this->current_modelInfo->orthogonalPlanesModels[i]->GetProperty()->SetLookupTable(ctf);
            }
        }

        // use default LUT
        else if(index == 0)
        {
            for(int i = 0; i<3; i++)
            {
                this->current_modelInfo->orthogonalPlanesModels[i]->GetProperty()->SetLookupTable(this->current_modelInfo->defaultLUT);
            }
        }
    }

    this->core()->render();
}

//------------------------[ Change visibility of model ]-----------------------\\

void TestPlugin::checkBoxVisibleChanged(bool checked)
{
    current_modelInfo->visible = checked;
    if(current_modelInfo->prop)
    {
        current_modelInfo->prop->SetVisibility(checked);
        this->core()->render();
    }
}

//------------------------[ Change rendering style of model ]-----------------------\\

void TestPlugin::comboBoxStyleChanged()
{
    // select model info matching the dataset
    int index = this->form->comboBoxStyle->currentIndex();

    current_modelInfo->renderStyle = index;
    if(current_modelInfo->prop != NULL)
    {
        switch(index)
        {
        case 0:
            current_modelInfo->prop->GetProperty()->SetRepresentationToSurface();
            break;

        case 1:
            current_modelInfo->prop->GetProperty()->SetRepresentationToWireframe();
            break;

        case 2:
            current_modelInfo->prop->GetProperty()->SetRepresentationToPoints();
            break;
        }
        this->core()->render();
    }
}

//------------------[ Click on the update button to start processing mesh ]-----------------------\\

void TestPlugin::buttonUpdateClicked()
{
    if(this->form->inputMinimumThreshold->value() > this->form->inputMaximumThreshold->value())
    {
        this->form->inputMinimumThreshold->setValue(this->form->inputMaximumThreshold->value());
        this->core()->render();
    }
    updateRenderingModels();
}

//------------------[ Change maximum threshold ]-----------------------\\

void TestPlugin::inputMaximumThresholdChanged(double value)
{
    current_modelInfo->maximumThreshold = value;
    bModelDirty = true;
    this->form->inputMinimumThreshold->setMaximum(value);
}

//------------------[ Change minimum threshold ]-----------------------\\

void TestPlugin::inputMinimumThresholdChanged(double value)
{
    current_modelInfo->minimumThreshold = value;
    bModelDirty = true;
    this->form->inputMaximumThreshold->setMinimum(value);
}

//------------------[ Change smoothing value ]-----------------------\\

void TestPlugin::inputSmoothingChanged(double value)
{
    current_modelInfo->smoothing = value;
    bModelDirty = true;
}

//------------------[ Change color ]-----------------------\

void TestPlugin::inputColorChanged(QString value)
{
    int color_hex = value.toInt(0,16);
    current_modelInfo->color[0] = ((color_hex >> 16) & 0xff) / 255.0; // red
    current_modelInfo->color[1] = ((color_hex >> 8) & 0xff) / 255.0; // green
    current_modelInfo->color[2] = (color_hex & 0xff) / 255.0; // blue
    current_modelInfo->colorString = value;
}

//------------------[ Change alpha ]-----------------------\

void TestPlugin::inputAlphaChanged(double value)
{
    current_modelInfo->alpha = value;
}

//------------------[ Change specularity ]-----------------------\

void TestPlugin::inputSpecularChanged(double value)
{
    current_modelInfo->specular = value;
}

//------------------[ Change mesh reduction ]-----------------------\

void TestPlugin::inputReductionChanged(double value)
{
    current_modelInfo->reduction = value;
    bModelDirty = true;
}

//---------------[ Change clipping plane slider in x direction ]-------------------\

void TestPlugin::horizontalSliderXChanged(int value)
{
    this->form->spinX->setValue(value);
    this->updateClippingPlaneSlider(0,value);
}

//---------------[ Change clipping plane slider in y direction ]-------------------\

void TestPlugin::horizontalSliderYChanged(int value)
{
    this->form->spinY->setValue(value);
    this->updateClippingPlaneSlider(1,value);
}

//---------------[ Change clipping plane slider in z direction ]-------------------\

void TestPlugin::horizontalSliderZChanged(int value)
{
    this->form->spinZ->setValue(value);
    this->updateClippingPlaneSlider(2,value);
}

//---------------[ Change clipping plane checkbox in x direction ]-------------------\

void TestPlugin::checkBoxXChanged(bool checked)
{
    this->updateClippingPlaneEnabled(0,checked);
}

//---------------[ Change clipping plane checkbox in y direction ]-------------------\

void TestPlugin::checkBoxYChanged(bool checked)
{
    this->updateClippingPlaneEnabled(1,checked);
}

//---------------[ Change clipping plane checkbox in z direction ]-------------------\

void TestPlugin::checkBoxZChanged(bool checked)
{
    this->updateClippingPlaneEnabled(2,checked);
}

//---------------[ Change clipping plane flip checkbox in x direction ]-------------------\

void TestPlugin::checkBoxFlipXChanged(bool checked)
{
    current_modelInfo->flipped[0] = checked;
    this->updateClippingPlaneSlider(0,current_modelInfo->clippingValues[0]);
}

//---------------[ Change clipping plane flip checkbox in y direction ]-------------------\

void TestPlugin::checkBoxFlipYChanged(bool checked)
{
    current_modelInfo->flipped[1] = checked;
    this->updateClippingPlaneSlider(1,current_modelInfo->clippingValues[1]);
}

//---------------[ Change clipping plane flip checkbox in z direction ]-------------------\

void TestPlugin::checkBoxFlipZChanged(bool checked)
{
    current_modelInfo->flipped[2] = checked;
    this->updateClippingPlaneSlider(2,current_modelInfo->clippingValues[2]);
}

//---------------[ Change clipping plane spin value in X direction ]-------------------\

void TestPlugin::spinXChanged(int value)
{
    this->form->horizontalSliderX->setValue(value);
    this->updateClippingPlaneSlider(0,value);
}

//---------------[ Change clipping plane spin value in Y direction ]-------------------\

void TestPlugin::spinYChanged(int value)
{
    this->form->horizontalSliderY->setValue(value);
    this->updateClippingPlaneSlider(1,value);
}

//---------------[ Change clipping plane spin value in Z direction ]-------------------\

void TestPlugin::spinZChanged(int value)
{
    this->form->horizontalSliderZ->setValue(value);
    this->updateClippingPlaneSlider(2,value);
}

//---------------[ Save 3D mesh ]-------------------\

void TestPlugin::buttonSaveMeshClicked()
{
    if(current_modelInfo->polydata != NULL)
    {
        QString fileName = QFileDialog::getSaveFileName(0, "Save 3D mesh","","VTK (*.vtk)");

		if(fileName.isEmpty())
			return;

        QByteArray ba = fileName.toLocal8Bit();
        char* fileName_c = ba.data();

        // export model
        vtkPolyDataWriter* polyDataWriter = vtkPolyDataWriter::New();
        polyDataWriter->SetInput(current_modelInfo->polydata);
        polyDataWriter->SetFileName(fileName_c);
        polyDataWriter->Write();
    }
}

void TestPlugin::checkBoxInvertClippingChanged(bool checked)
{
    if(current_modelInfo->extractPolyFunc != NULL)
    {
        current_modelInfo->invertClipping = checked;

        if(checked == false)
            current_modelInfo->extractPolyFunc->ExtractInsideOn();
        else
            current_modelInfo->extractPolyFunc->ExtractInsideOff();

        this->core()->render();
    }
}

void TestPlugin::checkBoxLargestComponentChanged(bool checked)
{
    current_modelInfo->selectLargestComponent = checked;
    bModelDirty = true;
}

void TestPlugin::checkBoxAlignPlanesToPickChanged(bool checked)
{
    current_modelInfo->alignPlanesToPick = checked;
}

void TestPlugin::buttonSetPointAClicked()
{
    setMeasuredPoint(0);
}

void TestPlugin::buttonSetPointBClicked()
{
    setMeasuredPoint(1);
}

void TestPlugin::setMeasuredPoint(int id)
{
    MeasuredPoint* point = this->measuredPointList.at(id);

    if(!point->set)
    {
        this->assembly->AddPart(point->sphere);
    }

    point->set = true;

    point->x = this->clickedPoint[0];
    point->y = this->clickedPoint[1];
    point->z = this->clickedPoint[2];

    point->sphere->SetPosition(point->x,point->y,point->z);

    if(id == 0)
    {
        this->form->inputXPointA->setValue(point->x);
        this->form->inputYPointA->setValue(point->y);
        this->form->inputZPointA->setValue(point->z);
    }
    else
    {
        this->form->inputXPointB->setValue(point->x);
        this->form->inputYPointB->setValue(point->y);
        this->form->inputZPointB->setValue(point->z);
    }

    calculateDistance();

    this->core()->render();
}

void TestPlugin::calculateDistance()
{
    MeasuredPoint* pointA = this->measuredPointList.at(0);
    MeasuredPoint* pointB = this->measuredPointList.at(1);

    if(!pointA->set || !pointB->set)
        return;

    double distance =   sqrt( (pointA->x - pointB->x)*(pointA->x - pointB->x) +
                        (pointA->y - pointB->y)*(pointA->y - pointB->y) +
                        (pointA->z - pointB->z)*(pointA->z - pointB->z) );
    QString labeltext = QString("Measured distance: %1").arg(distance,0,'f',4);
    this->form->measuredDistanceLabel->setText(labeltext);

    // renew line
    /*vtkSmartPointer<vtkLineSource> lineSource =
        vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(pointA->x,pointA->y,pointA->z);
    lineSource->SetPoint2(pointB->x,pointB->y,pointB->z)
    vtkSmartPointer<vtkPolyDataMapper> lineMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    lineMapper->SetInputConnection(lineSource->GetOutputPort());

    vtkActor* lineActor =
        vtkActor::New();
    lineActor->SetMapper(lineMapper);
    lineActor->GetProperty()->SetLineStipplePattern(0xFF00);

    this->assembly->AddPart(lineActor);
    if(measuredLine != NULL)
        this->assembly->RemovePart(measuredLine);
    measuredLine = lineActor;*/

    // remove old blobs
    QList<vtkActor*>::iterator i;
    for(i = this->depthElectrodeBlobs.begin(); i!=this->depthElectrodeBlobs.end(); i++)
    {
        this->assembly->RemovePart((*i));
    }
    this->depthElectrodeBlobs.clear();

    // create new blobs
    float stepSize = 4; // 1.5 mm depth electrode seperation
    int steps = ceil(distance/stepSize);

    vec3* line = new vec3;
    line->x = (pointA->x - pointB->x)/distance;
    line->y = (pointA->y - pointB->y)/distance;
    line->z = (pointA->z - pointB->z)/distance;

    for(int j = 0; j<steps; j++)
    {
        vtkSmartPointer<vtkSphereSource> diskSource =
            vtkSmartPointer<vtkSphereSource>::New();
        diskSource->SetRadius(2);

        vtkSmartPointer<vtkPolyDataMapper> diskMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
        diskMapper->SetInputConnection(diskSource->GetOutputPort());

        vtkActor* diskActor =
            vtkActor::New();
        diskActor->SetMapper(diskMapper);

        diskActor->SetPosition(-line->x * j * stepSize + pointA->x, -line->y * j * stepSize + pointA->y, -line->z * j * stepSize + pointA->z );

        if (this->currentElectrodesColor.isValid())
        {
            diskActor->GetProperty()->SetColor(this->currentElectrodesColor.redF(), this->currentElectrodesColor.greenF(), this->currentElectrodesColor.blueF());
        }

        this->assembly->AddPart(diskActor);
        this->depthElectrodeBlobs.append(diskActor);
    }
}

void TestPlugin::buttonSetLineColorClicked()
{
    double oldColorRGB[3];
    QColor oldColor;
    vtkActor* electrodeActor = depthElectrodeBlobs.at(0);
    electrodeActor->GetProperty()->GetColor(oldColorRGB);
    oldColor.setRgbF(oldColorRGB[0], oldColorRGB[1], oldColorRGB[2]);

    // Use a color dialog to get the new color
    this->currentElectrodesColor = QColorDialog::getColor(oldColor, 0);

    // set current electrodes to that color
    QList<vtkActor*>::iterator i;
    for(i = this->depthElectrodeBlobs.begin(); i!=this->depthElectrodeBlobs.end(); i++)
    {
        if (this->currentElectrodesColor.isValid())
        {
            (*i)->GetProperty()->SetColor(this->currentElectrodesColor.redF(), this->currentElectrodesColor.greenF(), this->currentElectrodesColor.blueF());
        }
    }
}

void TestPlugin::buttonSaveMeasurementClicked()
{
    this->depthElectrodeBlobs.clear();
}

///
///     vISTe communication
///

//-----------[ Returns visualization component as VTK object ]---------------\\
//
vtkProp * TestPlugin::getVtkProp()
{
    return this->assembly;
}

//-----------------[ Returns GUI component as Qt widget ]---------------\\
//
QWidget * TestPlugin::getGUI()
{
    return this->widget;
}

}

Q_EXPORT_PLUGIN2(libTestPlugin, bmia::TestPlugin)
