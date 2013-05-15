#include <DistanceUncertaintyPlugin.h>
#include <PolyDataVisualizationPlugin.h>

#include <gui/MetaCanvas/vtkMedicalCanvas.h>
#include <vtkImageDataToDistanceTransform.h>
#include <vtkParameterizePolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkPropAssembly.h>
#include <vtkProperty.h>
#include <vtkTexture.h>
#include <vtkLookupTable.h>
#include <vtkCamera.h>
#include <vtkMath.h>

#include <math.h>

#include <PBA3D/pba/pba3D.h>

#define PI 3.14159265358979323
#define ROUND(A) floor(A + 0.5)

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin::DistanceUncertaintyPlugin() :
		plugin::AdvancedPlugin( "DistanceUncertaintyPlugin" ), data::Consumer(), plugin::GUI()
	{
		_distanceWidget = new QDistanceWidget;
		_distanceWidget->setNumberOfSamples( 32 );
		_distanceWidget->setNumberOfRows( 128 );
		_distanceWidget->setNumberOfColumns( 256 );
		_distanceWidget->setNumberOfContours( 8 );
		_distanceWidget->setRiskRadius( 20 );
		_distanceWidget->setRiskRadiusEnabled( false );
		_distanceWidget->setContoursEnabled( false );
		_distanceWidget->setProjectionEnabled( false );

		this->connect( _distanceWidget, SIGNAL( computeStarted() ), this, SLOT( computeStarted() ) );
		this->connect( _distanceWidget->graph()->canvas(), SIGNAL( pointSelected( QPointF & ) ), this, SLOT( graphPointSelected( QPointF & ) ) );
		this->connect( _distanceWidget->graph()->canvas(), SIGNAL( rangeSelected( QPointF &, QPointF & ) ), this, SLOT( graphRangeSelected( QPointF &, QPointF &) ) );
		this->connect( _distanceWidget->map()->canvas(), SIGNAL( pointSelected( int ) ), this, SLOT( mapPointSelected( int ) ) );
		this->connect( _distanceWidget, SIGNAL( configurationChanged( QString ) ), this, SLOT( configChanged( QString ) ) );
		this->connect( _distanceWidget, SIGNAL( riskRadiusEnabled( bool ) ), this, SLOT( riskRadiusEnabled( bool ) ) );
		this->connect( _distanceWidget, SIGNAL( riskRadiusChanged( double ) ), this, SLOT( riskRadiusChanged( double ) ) );
		this->connect( _distanceWidget, SIGNAL( distanceContoursEnabled( bool, int ) ), this, SLOT( contoursEnabled( bool, int ) ) );
		this->connect( _distanceWidget, SIGNAL( distanceProjectionEnabled( bool ) ), this, SLOT( projectionEnabled( bool ) ) );
		this->connect( _distanceWidget, SIGNAL( automaticViewPointsEnabled( bool ) ), this, SLOT( automaticViewPointsEnabled( bool ) ) );
        this->connect( _distanceWidget, SIGNAL( colorLookupChanged( QString ) ), this, SLOT( colorLookupChanged( QString ) ) );
        this->connect( _distanceWidget, SIGNAL( computeSingleDTStarted()), this, SLOT( computeSingleDTStarted() ) );

		_pickerStyle = 0;
		_pickerStyleEventHandler = 0;

		_progressDialog.setText( "Computing distance information\nPlease wait..." );
		_progressDialog.setWindowTitle( "Information" );
		_progressDialog.setWindowModality( Qt::WindowModal );
		_progressDialog.hide();

	}

	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin::~DistanceUncertaintyPlugin()
	{
		delete _distanceWidget;
	}

	// PLUGIN METHODS

	///////////////////////////////////////////////////////////////////////////
	QWidget * DistanceUncertaintyPlugin::getGUI()
	{
		return _distanceWidget;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::dataSetAdded( data::DataSet * dataset )
	{
		if( dataset == 0 )
			return;

		QString name = dataset->getName();
		name.remove( 0, name.lastIndexOf( "/" ) + 1 );

		if( dataset->getKind() == "scalar volume" )
		{
			_distanceWidget->addDataset( name );
			_datasets.append( dataset->getVtkImageData() );
		}
		else if( dataset->getVtkPolyData() != NULL && dataset->getKind() != "fibers" )
		{
			_distanceWidget->addPolyDataset( name );
			_polyDatasets.append( dataset->getVtkPolyData() );
		}
		else {}

		if( ! _pickerStyle )
		{
			_pickerStyle = vtkInteractorStyleCellPicker::New();
			_pickerStyle->SetRenderer( this->getRenderer3D() );
			_pickerStyleEventHandler = new CellPickerEventHandler( this, _pickerStyle );

			this->getInteractor()->SetInteractorStyle( _pickerStyle );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	vtkRenderer * DistanceUncertaintyPlugin::getRenderer3D()
	{
		return this->fullCore()->canvas()->GetRenderer3D();
	}

	///////////////////////////////////////////////////////////////////////////
	vtkRenderWindowInteractor * DistanceUncertaintyPlugin::getInteractor()
	{
		return this->fullCore()->canvas()->GetInteractor();
	}

	///////////////////////////////////////////////////////////////////////////
	vtkInteractorObserver * DistanceUncertaintyPlugin::getInteractorStyle()
	{
		return this->getInteractor()->GetInteractorStyle();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::render()
	{
		this->fullCore()->render();
	}

	// SLOTS

    ///////////////////////////////////////////////////////////////////////////
    void DistanceUncertaintyPlugin::computeSingleDTStarted()
    {
        QString volumeName = _distanceWidget->selectedVolumeName();
        vtkImageData * volume = _datasets.at(_distanceWidget->selectedVolumeIndex());
        Q_ASSERT(volume);

        vtkImageDataToDistanceTransform * filter = vtkImageDataToDistanceTransform::New();
        filter->SetDistanceInverted(_distanceWidget->distanceInverted());
        filter->SetInput(volume);
        filter->SetThreshold(0);
        filter->Execute();

        vtkImageData * volumeDT = filter->GetOutput();
        Q_ASSERT(volumeDT);

        data::DataSet * dataset = new data::DataSet(
                volumeName.append( "Voxels" ), "scalar volume", volumeDT );
        this->core()->data()->addDataSet( dataset );

        filter->Delete();
    }

    ///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::computeStarted()
	{
		// Get selected tumor and fiber datasets
		vtkImageData * tumorData = _datasets.at( _distanceWidget->selectedTumorIndex() );
		Q_ASSERT( tumorData );
		vtkImageData * fiberData = _datasets.at( _distanceWidget->selectedFiberIndex() );
		Q_ASSERT( fiberData );

		// Check if we already have a configuration for the given tumor
		// and fiber data combination
		DistanceConfiguration * config = this->findDistanceConfiguration( tumorData, fiberData );
		if( config != NULL )
		{
			QMessageBox::warning( 0, "Attention", "Already have configuration for given tumor and fiber data" );
			return;
		}

		// Clear existing line and point marker widgets
		this->clearArrowWidgets();
		this->clearPointMarkerWidgets();

		// Get tumor and fiber dataset names
		QString tumorName = _distanceWidget->selectedTumorName();
		QString fiberName = _distanceWidget->selectedFiberName();

		// Create a new configuration and set tumor and fiber data
		config = new DistanceConfiguration( tumorName.append( fiberName ) );
		config->setTumorData( tumorData );
		config->setFiberData( fiberData );

		// Get tumor dimensions
		int dimensions[3];
		tumorData->GetDimensions( dimensions );
		int nx = dimensions[0];
		int ny = dimensions[1];
		int nz = dimensions[2];
		config->setDimensions( dimensions );
		std::cout << "computeStarted() dimensions = " << nx << " " << ny << " " << nz << std::endl;

		// Get tumor voxel spacing
		double spacing[3];
		tumorData->GetSpacing( spacing );
		double sx = spacing[0];
		double sy = spacing[1];
		double sz = spacing[2];
		config->setSpacing( spacing );
		std::cout << "computeStarted() spacing = " << sx << " " << sy << " " << sz << std::endl;

		// Compute tumor boundary voxel positions
		int ** tumorVoxelPositions = 0;
		int tumorNrVoxelPositions = 0;
		unsigned char * tumorVoxels = (unsigned char *) tumorData->GetScalarPointer();
		this->getBoundaryVoxelPositions( tumorVoxels, tumorVoxelPositions, tumorNrVoxelPositions, nx, ny, nz );
		config->setTumorVoxelPositions( tumorVoxelPositions, tumorNrVoxelPositions );
		std::cout << "computeStarted() tumorNrVoxelPositions = " << tumorNrVoxelPositions << std::endl;

		// Compute tumor centroid
		double tumorCentroid[3];
		this->computeCentroid( tumorVoxelPositions, tumorNrVoxelPositions, tumorCentroid, sx, sy, sz );
		config->setTumorCentroid( tumorCentroid );
		std::cout << "computeStarted() tumorCentroid = " << tumorCentroid[0] << " " << tumorCentroid[1] << " " << tumorCentroid[2] << std::endl;

		// Compute tumor min/max radius
		double tumorMinMaxRadius[2];
		this->computeMinMaxRadius( tumorVoxelPositions, tumorNrVoxelPositions, tumorCentroid, tumorMinMaxRadius, sx, sy, sz );
		config->setTumorMinMaxRadius( tumorMinMaxRadius[0], tumorMinMaxRadius[1] );
		std::cout << "computeStarted() tumorMinMaxRadius = " << tumorMinMaxRadius[0] << " " << tumorMinMaxRadius[1] << std::endl;

		// Compute fiber boundary voxel positions
		int ** fiberVoxelPositions = 0;
		int fiberNrVoxelPositions = 0;
		unsigned short * fiberVoxels = (unsigned short *) fiberData->GetScalarPointer();
		this->getBoundaryVoxelPositions( fiberVoxels, fiberVoxelPositions, fiberNrVoxelPositions, nx, ny, nz );
		config->setFiberVoxelPositions( fiberVoxelPositions, fiberNrVoxelPositions );
		std::cout << "computeStarted() fiberNrVoxelPositions = " << fiberNrVoxelPositions << std::endl;

		// Compute fiber centroid
		double fiberCentroid[3];
		this->computeCentroid( fiberVoxelPositions, fiberNrVoxelPositions, fiberCentroid, sx, sy, sz );
		config->setFiberCentroid( fiberCentroid );
		std::cout << "computeStarted() fiberCentroid = " << fiberCentroid[0] << " " << fiberCentroid[1] << " " << fiberCentroid[2] << std::endl;

		// Get tumor polydata and actor
		vtkPolyData * tumorPolyData = _polyDatasets.at( _distanceWidget->selectedTumorMeshIndex() );
		vtkParameterizePolyData * parameterizer = vtkParameterizePolyData::New();
		parameterizer->SetInput( tumorPolyData );
		parameterizer->SetCentroid( tumorCentroid );
		parameterizer->Execute();
		parameterizer->Delete();
		vtkActor * tumorActor = this->findActor( tumorPolyData );
		config->setTumorActor( tumorActor );

		// Keep track of minimum and maximum minimal distance. We need this to
		// scale the Y-axis of the minimal distance widget
		double minMinDist = 99999999.0;
		double maxMinDist = 0.0;
		double minMaxDist = 99999999.0;
		double maxMaxDist = 0.0;

		// Get rows and columns for the distance map
		int rows = _distanceWidget->numberOfRows();
		int columns = _distanceWidget->numberOfColumns();
		std::cout << "computeStarted() rows = " << rows << std::endl;
		std::cout << "computeStarted() columns = " << columns << std::endl;

		// Get threshold range and compute threshold step size
		int nrSamples = _distanceWidget->numberOfSamples();
		double * thresholdRange = fiberData->GetScalarRange();
		double thresholdStep = (thresholdRange[1] - thresholdRange[0]) / (nrSamples - 1);
		config->setThresholdRange( thresholdRange[0], thresholdRange[1] );
		std::cout << "computeStarted() nrSamples = " << nrSamples << std::endl;
		std::cout << "computeStarted() thresholdRange = " << thresholdRange[0] << " " << thresholdRange[1] << std::endl;
		std::cout << "computeStarted() thresholdStep = " << thresholdStep << std::endl;

		// Show progress dialog
		_progressDialog.show();

		// Compute distance map for each sample, except last one
		for( int i = 0; i < nrSamples - 1; ++i )
		{
			// Set current threshold
			double threshold = thresholdRange[0] + i * thresholdStep;

			// Compute distance transform
			vtkImageDataToDistanceTransform * filter = vtkImageDataToDistanceTransform::New();
			filter->SetInput( fiberData );
			filter->SetThreshold( threshold );
			filter->Execute();

			// Get voxels of distance transform
			vtkImageData * transform = filter->GetOutput();
			float * transformVoxels = (float *) transform->GetScalarPointer();

			// Get voxels of voronoi transform
			vtkImageData * voronoi = filter->GetOutputVoronoi();
			int * voronoiVoxels = (int *) voronoi->GetScalarPointer();

			// Compute minimal and maximum distance position
			double minDist, maxDist, minDistPos1[3], minDistPos2[3];
			this->computeMinMaxDistancePosition( tumorVoxelPositions, tumorNrVoxelPositions, transformVoxels,
				voronoiVoxels, minDist, minDistPos1, minDistPos2, maxDist, nx, ny, nz, sx, sy, sz );
			config->getMinimumDistances().append( minDist );
			config->getMaximumDistances().append( maxDist );
			config->getMinimumDistanceStartPoints().append( minDistPos1 );
			config->getMinimumDistanceEndPoints().append( minDistPos2 );

			// Store threshold and minimal distance in list of points
			QPointF thresholdDistancePoint;
			thresholdDistancePoint.setX( threshold );
			thresholdDistancePoint.setY( minDist );
			config->getThresholdDistancePoints().append( thresholdDistancePoint );
			std::cout << "  computeStarted() thresholdDistancePoint = " << threshold << " " << minDist << std::endl;

			// Update minimum and maximum distance
			if( minDist > maxMinDist )
				maxMinDist = minDist;
			if( minDist < minMinDist )
				minMinDist = minDist;
			if( maxDist > maxMaxDist )
				maxMaxDist = maxDist;
			if( maxDist < minMaxDist )
				minMaxDist = maxDist;

			// Delete distance transform filter
			filter->Delete();
		}

		_progressDialog.hide();

		// Check that minimum and maximum distance are not equal. If so,
		// add a small offset to minimal distance to get maximal distance
		if( minMinDist == maxMinDist )
			maxMinDist = minMinDist + 1.0;

		// Add last threshold/distance point to the list
		QPointF lastThresholdDistancePoint;
		lastThresholdDistancePoint.setX( thresholdRange[1] );
		lastThresholdDistancePoint.setY( config->getThresholdDistancePoints().last().y() );
		config->getThresholdDistancePoints().append( lastThresholdDistancePoint );

		// Set additional properties of configuration
		config->setMinimumDistanceRange( minMinDist, maxMinDist );
		config->setMaximumDistanceRange( minMaxDist, maxMaxDist );
		config->setRiskRadius( _distanceWidget->riskRadius() );
		config->setRiskRadiusEnabled( _distanceWidget->riskRadiusEnabled() );
		config->setContoursEnabled( _distanceWidget->contoursEnabled() );
		config->setNumberOfContours( _distanceWidget->numberOfContours() );
		config->setProjectionEnabled( _distanceWidget->projectionEnabled() );

		std::cout << "computeStarted() minMinDist maxMinDist = " << minMinDist << " " << maxMinDist << std::endl;
		std::cout << "computeStarted() minMaxDist maxMaxDist = " << minMaxDist << " " << maxMaxDist << std::endl;
		std::cout << "computeStarted() riskRadius = " << _distanceWidget->riskRadius() << std::endl;
		std::cout << "computeStarted() riskRadiusEnabled = " << (_distanceWidget->riskRadiusEnabled() ? "true" : "false") << std::endl;
		std::cout << "computeStarted() contoursEnabled = " << (_distanceWidget->contoursEnabled() ? "true" : "false") << std::endl;
		std::cout << "computeStarted() numberOfContours = " << _distanceWidget->numberOfContours() << std::endl;
		std::cout << "computeStarted() projectionEnabled = " << (_distanceWidget->projectionEnabled() ? "true" : "false") << std::endl;

		// Compute default sample index
		int defaultIdx = (int) nrSamples / 2;
		double defaultDist = config->getThresholdDistancePoints().at( defaultIdx ).y();
		double defaultThreshold = thresholdRange[0] + defaultIdx * thresholdStep;
		double * defaultStartPoint = config->getMinimumDistanceStartPoints().at( defaultIdx );
		double * defaultEndPoint = config->getMinimumDistanceEndPoints().at( defaultIdx );

		std::cout << "computeStarted() defaultIdx = " << defaultIdx << std::endl;
		std::cout << "computeStarted() defaultDist = " << defaultDist << std::endl;
		std::cout << "computeStarted() defaultStartPoint = " << defaultStartPoint[0] << " " << defaultStartPoint[1] << " " << defaultStartPoint[2] << std::endl;
		std::cout << "computeStarted() defaultEndPoint = " << defaultEndPoint[0] << " " << defaultEndPoint[1] << " " << defaultEndPoint[2] << std::endl;

		// Add default threshold and distance as selected value in configuration
		config->setSelectedThreshold( defaultThreshold );
		config->setSelectedDistance( defaultDist );

		// Update distance graph widget
		_distanceWidget->graph()->canvas()->setRangeX( config->getThresholdMin(), config->getThresholdMax() );
		_distanceWidget->graph()->canvas()->setRangeY( config->getMinimumDistanceMin(), config->getMinimumDistanceMax() );
		_distanceWidget->graph()->canvas()->setPoints( config->getThresholdDistancePoints() );

		// Update distance map widget
		_distanceWidget->map()->canvas()->setMapRange( config->getMinimumDistanceMin(), config->getMaximumDistanceMax() );
		_distanceWidget->map()->canvas()->setRiskRadius( config->getRiskRadius() );
		_distanceWidget->map()->canvas()->setRiskRadiusEnabled( config->isRiskRadiusEnabled() );
		_distanceWidget->map()->canvas()->setProjectionEnabled( config->isProjectionEnabled() );
		_distanceWidget->map()->canvas()->setContoursEnabled( config->isContoursEnabled() );
		_distanceWidget->map()->canvas()->setNumberOfContours( config->getNumberOfContours() );

		// Set configuration as current configuration
		_configurations.append( config );
		_distanceWidget->addConfiguration( config->getName() );
		_currentConfig = config;

		// Point camera directly at tumor centroid
		vtkCamera * camera = this->getRenderer3D()->GetActiveCamera();
		camera->SetFocalPoint( tumorCentroid );

		// Select default threshold
		QPointF graphPoint( defaultThreshold, defaultDist );
		this->graphPointSelected( graphPoint );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::mapPointSelected( int voxelIdx )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		int nx = _currentConfig->getDimensionX();
		int ny = _currentConfig->getDimensionY();
		int nz = _currentConfig->getDimensionZ();

		// Compute XYZ coordinate from voxel index
        int z = (int) floor( (double)  voxelIdx / (nx * ny) );
        int y = (int) floor( (double) (voxelIdx - z * nx * ny) / nx );
        int x = (int) floor( (double)  voxelIdx - z * nx * ny - y * nx );

		// Remove existing point markers from scene
		this->clearPointMarkerWidgets();

		// Get tumor voxel position
		double P[3];
		P[0] = _currentConfig->getSpacingX() * x;
		P[1] = _currentConfig->getSpacingY() * y;
		P[2] = _currentConfig->getSpacingZ() * z;

		// Add point marker at position
		this->addPointMarkerWidget( P );

		// Set current map point in current config
		_currentConfig->setSelectedPointMarkerPosition( P );

		// Update camera position to provide optimal view of point marker
		if( _distanceWidget->automaticViewPointsEnabled() )
		{
			this->updateViewPoint( P );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::graphPointSelected( QPointF & point )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		// Remove line widgets from scene
		this->clearArrowWidgets();

		// Get selected threshold
		double thresholdRange[2];
		thresholdRange[0] = _currentConfig->getThresholdMin();
		thresholdRange[1] = _currentConfig->getThresholdMax();
		double threshold = (double) point.x();
		std::cout << "graphPointSelected() thresholdRange = " << thresholdRange[0] << " " << thresholdRange[1] << std::endl;
		std::cout << "graphPointSelected() threshold = " << threshold << std::endl;

		// Compute distance transform for selected threshold
		vtkImageDataToDistanceTransform * filter = vtkImageDataToDistanceTransform::New();
		filter->SetInput( _currentConfig->getFiberData() );
		filter->SetThreshold( threshold );
		filter->Execute();

		// Get transform voxels
		vtkImageData * transform = filter->GetOutput();
		float * transformVoxels = (float *) transform->GetScalarPointer();

		// Get voronoi voxels
		vtkImageData * voronoi = filter->GetOutputVoronoi();
		int * voronoiVoxels = (int *) voronoi->GetScalarPointer();
		_currentConfig->setSelectedVoronoiData( voronoi );

		// Get tumor voxels and voxel positions
		vtkImageData * tumorData = _currentConfig->getTumorData();
		unsigned char * tumorVoxels = (unsigned char *) tumorData->GetScalarPointer();
		int ** tumorVoxelPositions = _currentConfig->getTumorVoxelPositions();
		int tumorNrVoxelPositions = _currentConfig->getNumberOfTumorVoxelPositions();

		// Set selected color scale
		_currentConfig->setSelectedColorScaleName( _distanceWidget->selectedColorLookupTableName() );

		// Get transform dimensions
		int dims[3], nx, ny, nz;
		transform->GetDimensions( dims );
		nx = dims[0];
		ny = dims[1];
		nz = dims[2];
		std::cout << "graphPointSelected() nx ny nz = " << nx << " " << ny << " " << nz << std::endl;

		// Get transform voxel spacing
		double spacing[3], sx, sy, sz;
		transform->GetSpacing( spacing );
		sx = spacing[0];
		sy = spacing[1];
		sz = spacing[2];
		std::cout << "graphPointSelected() sx sy sz = " << sx << " " << sy << " " << sz << std::endl;

		// Compute minimum distance
		double minDist, maxDist, minDistPos1[3], minDistPos2[3];
		this->computeMinMaxDistancePosition( tumorVoxelPositions, tumorNrVoxelPositions, transformVoxels,
				voronoiVoxels, minDist, minDistPos1, minDistPos2, maxDist, nx, ny, nz, sx, sy, sz );
		std::cout << "graphPointSelected() minDist maxDist = " << minDist << " " << maxDist << std::endl;
		std::cout << "graphPointSelected() minDistPos1 = " << minDistPos1[0] << " " << minDistPos1[1] << " " << minDistPos1[2] << std::endl;
		std::cout << "graphPointSelected() minDistPos2 = " << minDistPos2[0] << " " << minDistPos2[1] << " " << minDistPos2[2] << std::endl;

		// Get rows and columns
		int rows = _distanceWidget->numberOfRows();
		int columns = _distanceWidget->numberOfColumns();
		std::cout << "graphPointSelected() rows columns = " << rows << " " << columns << std::endl;

		// Get tumor centroid
		double tumorCentroid[3];
		tumorCentroid[0] = _currentConfig->getTumorCentroidX();
		tumorCentroid[1] = _currentConfig->getTumorCentroidY();
		tumorCentroid[2] = _currentConfig->getTumorCentroidZ();
		double tumorMinRadius = _currentConfig->getTumorMinRadius();

		// Get fiber centroid
		double fiberCentroid[3];
		fiberCentroid[0] = _currentConfig->getFiberCentroidX();
		fiberCentroid[1] = _currentConfig->getFiberCentroidY();
		fiberCentroid[2] = _currentConfig->getFiberCentroidZ();

		// Compute theta and phi of vector between tumor and fiber centroids
        double V[3], R, deltaTheta, deltaPhi;
		V[0] = minDistPos1[0] - tumorCentroid[0];
		V[1] = minDistPos1[1] - tumorCentroid[1];
		V[2] = minDistPos1[2] - tumorCentroid[2];
		this->normalizeVector( V );
        this->cartesianToSpherical( V, R, deltaTheta, deltaPhi );

		int * indexMap = new int[rows * columns];
		double * distanceMap = new double[rows * columns];
		this->computeDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMap, indexMap, rows, columns,
            tumorCentroid,	tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz );

		// Add texture to tumor actor
		vtkTexture * texture = this->buildTexture( distanceMap, 128, 256 );
		_currentConfig->applyTexture( texture );
		_currentConfig->updateTextureLookupTable( texture );
		texture->Delete();

        std::cout << "graphPointSelected() minimal distance = " << minDist << std::endl;
        std::cout << "graphPointSelected() minimal distance position = " << minDistPos1[0] << " " << minDistPos1[1] << " " << minDistPos1[2] << std::endl;

        QPoint ppp = this->getMapPosition(minDistPos1, tumorCentroid, rows, columns, false);
        std::cout << "P(" << ppp.x() << "," << ppp.y() << ")" << std::endl;

        // Recompute distance map for the map widget
		if( _currentConfig->isProjectionEnabled() )
		{
			this->computeSinusoidalDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMap, indexMap, rows, columns,
                tumorCentroid,tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz, deltaTheta, deltaPhi );

            QPoint pp = this->getMapPosition(minDistPos1, tumorCentroid, rows, columns, true);
            std::cout << "Pproj(" << pp.x() << "," << pp.y() << ")" << std::endl;
        }

        QList<double> distances;
        QList<QPoint> positions;

        // Compute magnitudes and positions of minimal distances across the
        // probability interval specified
        if( _distanceWidget->uncertaintyEnabled() && _distanceWidget->probabilityInterval() > 0.0 )
        {
            double interval = _distanceWidget->probabilityInterval();

            // Compute +/- offsets as percentage of threshold range
            double thresholdMin = threshold - interval * (thresholdRange[1] - thresholdRange[0]);
            double thresholdMax = threshold + interval * (thresholdRange[1] - thresholdRange[0]);
            double step = (thresholdMax - thresholdMin) / 10.0;
            double value = thresholdMin;

            for(int i = 0; i < 10; ++i)
            {

                vtkImageDataToDistanceTransform * filterTmp = vtkImageDataToDistanceTransform::New();
                filterTmp->SetInput(_currentConfig->getFiberData());
                filterTmp->SetThreshold(value);
                filterTmp->Execute();

                vtkImageData * tmpTransform =  filterTmp->GetOutput();
                float * tmpTransformVoxels = (float *) tmpTransform->GetScalarPointer();
                vtkImageData * tmpVoronoi = filterTmp->GetOutputVoronoi();
                int * tmpVoronoiVoxels = (int *) tmpVoronoi->GetScalarPointer();

                double tmpMinDist, tmpMaxDist, tmpMinDistPos1[3], tmpMinDistPos2[3];
                this->computeMinMaxDistancePosition( tumorVoxelPositions, tumorNrVoxelPositions, tmpTransformVoxels,
                        tmpVoronoiVoxels, tmpMinDist, tmpMinDistPos1, tmpMinDistPos2, tmpMaxDist, nx, ny, nz, sx, sy, sz );

                // Translate minimal distance position to 2D map position
                double tmpV[3], tmpR, tmpTheta, tmpPhi;
                tmpV[0] = tmpMinDistPos1[0] - tumorCentroid[0];
                tmpV[1] = tmpMinDistPos1[1] - tumorCentroid[1];
                tmpV[2] = tmpMinDistPos1[2] - tumorCentroid[2];
                this->normalizeVector( tmpV );
                this->cartesianToSpherical( tmpV, tmpR, tmpTheta, tmpPhi );

                // Back-project (theta,phi) to map coordinates depending on whether cylindrical
                // or sinusoidal projection was enabled
                if(_currentConfig->isProjectionEnabled())
                {
                    QPoint point(
                        (int) columns * (tmpTheta * cos(tmpPhi) + PI) / (2*PI),
                        (int) rows * (tmpPhi + PI/2) / PI);
                    positions.append(point);
                }
                else
                {
                    QPoint point(
                        (int) columns * (tmpTheta + PI) / (2*PI),
                        (int) rows * (tmpPhi + PI/2) / PI);
                    positions.append(point);
                    //std::cout << "P(" << point.x() << "," << point.y() << ")" << std::endl;
                }

                QPoint point = this->getMapPosition(tmpMinDistPos1, tumorCentroid,
                        rows, columns, _currentConfig->isProjectionEnabled());
                //std::cout << "P(" << point.x() << "," << point.y() << ")" << std::endl;

                positions.append(point);
                distances.append(tmpMinDist);

                value += step;

                filterTmp->Delete();
            }
        }

		// Update distance widget graph and map
		_distanceWidget->graph()->canvas()->setThreshold( threshold );
		_distanceWidget->graph()->canvas()->setDistance( minDist );
		_distanceWidget->map()->canvas()->setMinimalDistance( minDist );
		_distanceWidget->map()->canvas()->setDistanceMap( distanceMap, rows, columns );
		_distanceWidget->map()->canvas()->setIndexMap( indexMap, rows, columns );
        _distanceWidget->map()->canvas()->setMinimalDistances(distances, positions);

		// Clean up maps
		delete [] distanceMap;
		delete [] indexMap;

		// Update selected items of current configuration. This will store the
		// threshold/distance point for later retrieval
		_currentConfig->setSelectedDistance( minDist );
		_currentConfig->setSelectedThreshold( threshold );

		// Update raycast mapper iso-value
		RayCastVolumeMapper * mapper = this->findVolumeMapper();
		if( mapper )
		{
			// Check if we're raycasting the fiber tract volume. If not,
			// we should not set the isovalue
			if( _distanceWidget->updateIsoValueEnabled() )
			{
				mapper->setIsoValue( threshold );
			}
		}

		// Add line widget to scene
		this->addArrowWidget( minDistPos1, minDistPos2, minDist );

		// Delete filter
		filter->Delete();
	}

    ///////////////////////////////////////////////////////////////////////////
    QPoint DistanceUncertaintyPlugin::getMapPosition(double position[3], double centroid[3],
            int rows, int columns, bool projection)
    {
        // Translate minimal distance position to 2D map position
        double tmpV[3], tmpR, tmpTheta, tmpPhi;
        tmpV[0] = position[0] - centroid[0];
        tmpV[1] = position[1] - centroid[1];
        tmpV[2] = position[2] - centroid[2];
        this->normalizeVector( tmpV );
        this->cartesianToSpherical( tmpV, tmpR, tmpTheta, tmpPhi );

        // Back-project (theta,phi) to map coordinates depending on whether cylindrical
        // or sinusoidal projection was enabled
        if(projection)
        {
            return QPoint(
                (int) columns * (tmpTheta * cos(tmpPhi) + PI) / (2*PI),
                (int) rows * (tmpPhi + PI/2) / PI);
        }

        return QPoint(
            (int) columns * (tmpTheta + PI) / (2*PI),
            (int) rows * (tmpPhi + PI/2) / PI);
    }

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::graphRangeSelected( QPointF & point1, QPointF & point2 )
	{
        if( _currentConfig == 0 )
            return;
        Q_ASSERT( _currentConfig->isValid() );

        // Remove line widgets from scene
        this->clearArrowWidgets();

        // Get selected threshold
        double thresholdRange[2];
        thresholdRange[0] = _currentConfig->getThresholdMin();
        thresholdRange[1] = _currentConfig->getThresholdMax();

        // Get tumor voxels and voxel positions
        vtkImageData * tumorData = _currentConfig->getTumorData();
        unsigned char * tumorVoxels = (unsigned char *) tumorData->GetScalarPointer();
        int ** tumorVoxelPositions = _currentConfig->getTumorVoxelPositions();
        int tumorNrVoxelPositions = _currentConfig->getNumberOfTumorVoxelPositions();

        // Get rows and columns
        int rows = _distanceWidget->numberOfRows();
        int columns = _distanceWidget->numberOfColumns();

        // Get tumor centroid
        double tumorCentroid[3];
        tumorCentroid[0] = _currentConfig->getTumorCentroidX();
        tumorCentroid[1] = _currentConfig->getTumorCentroidY();
        tumorCentroid[2] = _currentConfig->getTumorCentroidZ();
        double tumorMinRadius = _currentConfig->getTumorMinRadius();

        // Get fiber centroid
        double fiberCentroid[3];
        fiberCentroid[0] = _currentConfig->getFiberCentroidX();
        fiberCentroid[1] = _currentConfig->getFiberCentroidY();
        fiberCentroid[2] = _currentConfig->getFiberCentroidZ();

        // Find out which point is associated with minimum threshold, and
        // which with the maximum threshold. Point A will be the point for which
        // we show the distance map. Point B will be the point for we show the
        // uncertainty in the risk area
        QPointF pointA = point1.x() > point2.x() ? point1 : point2;
        QPointF pointB = point1.x() > point2.x() ? point2 : point1;

        int dims[3], nx, ny, nz;
        double spacing[3], sx, sy, sz;
        int * indexMap = new int[rows * columns];
        int * indexMapSinus = new int[rows * columns];
        double * distanceMapA = new double[rows * columns];
        double * distanceMapSinusA = new double[rows * columns];
        double * distanceMapB = new double[rows * columns];
        double * distanceMapSinusB = new double[rows * columns];
        double minimalDistance = 0.0;

        // Compute distance map for point A threshold
        {
            vtkImageDataToDistanceTransform * filter = vtkImageDataToDistanceTransform::New();
            filter->SetInput( _currentConfig->getFiberData() );
            filter->SetThreshold( pointA.x() );
            filter->Execute();

            // Get transform voxels
            vtkImageData * transform = filter->GetOutput();
            float * transformVoxels = (float *) transform->GetScalarPointer();

            // Get voronoi voxels
            vtkImageData * voronoi = filter->GetOutputVoronoi();
            int * voronoiVoxels = (int *) voronoi->GetScalarPointer();

            // Get transform dimensions
            transform->GetDimensions( dims );
            nx = dims[0];
            ny = dims[1];
            nz = dims[2];

            // Get transform voxel spacing
            transform->GetSpacing( spacing );
            sx = spacing[0];
            sy = spacing[1];
            sz = spacing[2];

            // Compute minimum distance
            double minDist, maxDist, minDistPos1[3], minDistPos2[3];
            this->computeMinMaxDistancePosition( tumorVoxelPositions, tumorNrVoxelPositions, transformVoxels,
                    voronoiVoxels, minDist, minDistPos1, minDistPos2, maxDist, nx, ny, nz, sx, sy, sz );
            minimalDistance = minDist;

            // Compute theta and phi of vector between (0,0,1) and minimal distance point
            double V[3], R, deltaTheta, deltaPhi;
            V[0] = minDistPos1[0] - tumorCentroid[0];
            V[1] = minDistPos1[1] - tumorCentroid[1];
            V[2] = minDistPos1[2] - tumorCentroid[2];
            this->normalizeVector( V );
            this->cartesianToSpherical( V, R, deltaTheta, deltaPhi );

            // Compute both normal and sinusoidal distance map
            this->computeDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMapA, indexMap, rows, columns,
                tumorCentroid,tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz );
            this->computeSinusoidalDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMapSinusA, indexMapSinus, rows, columns,
                tumorCentroid,tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz, deltaTheta, deltaPhi );
        }

        // Compute distance map for point B threshold
        {
            vtkImageDataToDistanceTransform * filter = vtkImageDataToDistanceTransform::New();
            filter->SetInput( _currentConfig->getFiberData() );
            filter->SetThreshold( pointB.x() );
            filter->Execute();

            // Get transform voxels
            vtkImageData * transform = filter->GetOutput();
            float * transformVoxels = (float *) transform->GetScalarPointer();

            // Get voronoi voxels
            vtkImageData * voronoi = filter->GetOutputVoronoi();
            int * voronoiVoxels = (int *) voronoi->GetScalarPointer();

            // Compute minimum distance
            double minDist, maxDist, minDistPos1[3], minDistPos2[3];
            this->computeMinMaxDistancePosition( tumorVoxelPositions, tumorNrVoxelPositions, transformVoxels,
                    voronoiVoxels, minDist, minDistPos1, minDistPos2, maxDist, nx, ny, nz, sx, sy, sz );

            // Compute theta and phi of vector between (0,0,1) and minimal distance point
            double V[3], R, deltaTheta, deltaPhi;
            V[0] = minDistPos1[0] - tumorCentroid[0];
            V[1] = minDistPos1[1] - tumorCentroid[1];
            V[2] = minDistPos1[2] - tumorCentroid[2];
            this->normalizeVector( V );
            this->cartesianToSpherical( V, R, deltaTheta, deltaPhi );

            // Compute both normal and sinusoidal distance map
            this->computeDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMapA, indexMap, rows, columns,
                tumorCentroid,tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz );
            this->computeSinusoidalDistanceMap( tumorVoxels, transformVoxels, voronoiVoxels, distanceMapSinusA, indexMapSinus, rows, columns,
                tumorCentroid,tumorMinRadius, 0.5, nx, ny, nz, sx, sy, sz, deltaTheta, deltaPhi );
        }

        // Add texture to tumor actor
        vtkTexture * texture = this->buildTexture( distanceMapA, 128, 256 );
        _currentConfig->applyTexture( texture );
        _currentConfig->updateTextureLookupTable( texture );
        texture->Delete();

        // Update distance widget graph and map
        _distanceWidget->graph()->canvas()->setThreshold( pointA.x() );
        _distanceWidget->graph()->canvas()->setDistance( minimalDistance );
        _distanceWidget->map()->canvas()->setMinimalDistance( minimalDistance );
        _distanceWidget->map()->canvas()->setDistanceMap( distanceMapA, distanceMapB, rows, columns );
        _distanceWidget->map()->canvas()->setIndexMap( indexMap, rows, columns );

        if( _currentConfig->isProjectionEnabled() )
        {
            _distanceWidget->map()->canvas()->setDistanceMap( distanceMapSinusA, distanceMapSinusB, rows, columns );
            _distanceWidget->map()->canvas()->setIndexMap( indexMapSinus, rows, columns );
        }

        // Clean up maps
        delete [] distanceMapA;
        delete [] distanceMapSinusA;
        delete [] distanceMapB;
        delete [] distanceMapSinusB;
        delete [] indexMap;

        // Update selected items of current configuration. This will store the
        // threshold/distance point for later retrieval
        _currentConfig->setSelectedDistance( minimalDistance );
        _currentConfig->setSelectedThreshold( pointA.x() );
    }

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::configChanged( QString name )
	{
		// Get selected configuration
		DistanceConfiguration * config = this->findDistanceConfiguration( name );
		Q_ASSERT( config );

		// Get rows and columns
		int rows = _distanceWidget->numberOfRows();
		int columns = _distanceWidget->numberOfColumns();

		// Update distance graph widget
		_distanceWidget->graph()->canvas()->setRangeX( config->getThresholdMin(), config->getThresholdMax() );
		_distanceWidget->graph()->canvas()->setRangeY( config->getMinimumDistanceMin(), config->getMinimumDistanceMax() );
		_distanceWidget->graph()->canvas()->setPoints( config->getThresholdDistancePoints() );
		_distanceWidget->graph()->canvas()->setThreshold( config->getSelectedThreshold() );
		_distanceWidget->graph()->canvas()->setDistance( config->getSelectedDistance() );

		// Update distance map widget
		_distanceWidget->map()->canvas()->setMapRange( config->getMinimumDistanceMin(), config->getMaximumDistanceMax() );

		// Update distance widget
		_distanceWidget->setRiskRadius( config->getRiskRadius() );
		_distanceWidget->setRiskRadiusEnabled( config->isRiskRadiusEnabled() );
		_distanceWidget->setProjectionEnabled( config->isProjectionEnabled() );
		_distanceWidget->setContoursEnabled( config->isContoursEnabled() );
		_distanceWidget->setNumberOfContours( config->getNumberOfContours() );

		double threshold = config->getSelectedThreshold();
		double distance  = config->getSelectedDistance();
		std::cout << "configChanged() threshold = " << threshold << ", distance = " << distance << std::endl;

		// Set as current config
		_currentConfig = config;

		// Trigger graph point selected event
		QPointF p( _currentConfig->getSelectedThreshold(), _currentConfig->getSelectedDistance() );
		this->graphPointSelected( p );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::riskRadiusEnabled( bool enabled )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		_currentConfig->setRiskRadiusEnabled( enabled );
		_currentConfig->updateTexture();

		this->fullCore()->render();
	}

    ///////////////////////////////////////////////////////////////////////////
    void DistanceUncertaintyPlugin::riskRadiusUncertaintyEnabled( bool enabled )
    {
        if( _currentConfig == 0 )
            return;
        Q_ASSERT( _currentConfig->isValid() );

        _currentConfig->setRiskRadiusUncertaintyEnabled( enabled );
        _currentConfig->updateTexture();

        this->fullCore()->render();
    }

    ///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::riskRadiusChanged( double radius )
	{
		if( _distanceWidget->riskRadiusEnabled() )
		{
			if( _currentConfig == 0 )
				return;
			Q_ASSERT( _currentConfig->isValid() );

			_currentConfig->setRiskRadius( radius );
			_currentConfig->updateTexture();

			this->fullCore()->render();
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::contoursEnabled( bool enabled, int numberOfContours )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		_currentConfig->setContoursEnabled( enabled );
		_currentConfig->setNumberOfContours( numberOfContours );
		_currentConfig->updateTexture();

		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::projectionEnabled( bool enabled )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		_currentConfig->setProjectionEnabled( enabled );
		double threshold = _currentConfig->getSelectedThreshold();
		double distance  = _currentConfig->getSelectedDistance();

		QPointF graphPoint( threshold, distance );
		this->graphPointSelected( graphPoint );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::automaticViewPointsEnabled( bool enabled )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		if( enabled )
		{
			double P[3];
			P[0] = _currentConfig->getSelectedPointMarkerPositionX();
			P[1] = _currentConfig->getSelectedPointMarkerPositionY();
			P[2] = _currentConfig->getSelectedPointMarkerPositionZ();

			if( P[0] > -1 && P[1] > -1 && P[2] > -1 )
			{
				this->updateViewPoint( P );
			}
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::updateIsoValueEnabled( bool enabled )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		double threshold = _currentConfig->getSelectedThreshold();
		double distance  = _currentConfig->getSelectedDistance();

		QPointF graphPoint( threshold, distance );
		this->graphPointSelected( graphPoint );
	}

    ///////////////////////////////////////////////////////////////////////////
    void DistanceUncertaintyPlugin::colorLookupChanged( QString lookup )
    {
        if(_currentConfig == 0)
            return;
        Q_ASSERT(_currentConfig->isValid());
        QPointF point = QPointF(_currentConfig->getSelectedThreshold(), _currentConfig->getSelectedDistance());
        this->graphPointSelected(point);
    }

    // HELPER METHODS

	///////////////////////////////////////////////////////////////////////////
	template< class T >
	void DistanceUncertaintyPlugin::getBoundaryVoxelPositions( T * voxels, int **& voxelPositions,
		int & nrPositions, int nx, int ny, int nz )
	{
		std::vector< int * > positions;

		for( int z = 0; z < nz; ++z )
		{
			for( int y = 0; y < ny; ++y )
			{
				for( int x = 0; x < nx; ++x )
				{
					int idx = z * nx * ny + y * nx + x;
					if( voxels[idx] )
					{
						if( (! voxels[z * nx * ny + y * nx + (x+1)]) ||
							(! voxels[z * nx * ny + y * nx + (x-1)]) ||
							(! voxels[z * nx * ny + (y+1) * nx + x]) ||
							(! voxels[z * nx * ny + (y-1) * nx + x]) ||
							(! voxels[(z+1) * nx * ny + y * nx + x]) ||
							(! voxels[(z-1) * nx * ny + y * nx + x]) )
						{
							int * position = new int[3];
							position[0] = x;
							position[1] = y;
							position[2] = z;
							positions.push_back( position );
						}
					}
				}
			}
		}

		voxelPositions = new int*[positions.size()];
		for( int i = 0; i < positions.size(); ++i )
		{
			int * position = positions.at( i );
			voxelPositions[i] = new int[3];
			voxelPositions[i][0] = position[0];
			voxelPositions[i][1] = position[1];
			voxelPositions[i][2] = position[2];
		}

		nrPositions = positions.size();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::computeCentroid( int ** positions, int nrPositions,
		double centroid[3], double sx, double sy, double sz )
	{
		double x = 0.0;
		double y = 0.0;
		double z = 0.0;

		for( int i = 0; i < nrPositions; ++i )
		{
			x += positions[i][0];
			y += positions[i][1];
			z += positions[i][2];
		}

		centroid[0] = sx * x / nrPositions;
		centroid[1] = sy * y / nrPositions;
		centroid[2] = sz * z / nrPositions;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::computeMinMaxRadius( int ** voxelPositions, int nrPositions,
		double centroid[3], double minMaxRadius[2], double sx, double sy, double sz  )
	{
		double minRadius = 99999999.0f;
		double maxRadius = 0.0f;

		for( int i = 0; i < nrPositions; ++i )
		{
			double dx = (sx * voxelPositions[i][0]) - centroid[0];
			double dy = (sy * voxelPositions[i][1]) - centroid[1];
			double dz = (sz * voxelPositions[i][2]) - centroid[2];

			float radius = sqrt( dx*dx + dy*dy + dz*dz );

			if( radius < minRadius )
				minRadius = radius;
			if( radius > maxRadius )
				maxRadius = radius;
		}

		minMaxRadius[0] = minRadius;
		minMaxRadius[1] = maxRadius;
	}

	///////////////////////////////////////////////////////////////////////////
	template< class T >
	void DistanceUncertaintyPlugin::computeDistanceFromSphericalCoordinates( T * voxels, float * transform, int * voronoi, double centroid[3],
		double minRadius, double threshold, int nx, int ny, int nz, double sx, double sy, double sz,
		double theta, double phi, double & distance, int & voxelIndex )
	{
		// Compute unit vector along direction of theta and phi
		double unitVector[3];
		this->sphericalToCartesian( 1.0, theta, phi, unitVector );

		// Define step vector for sampling along ray
		float step = 0.5f;
		float stepVector[3];
		stepVector[0] = step * unitVector[0];
		stepVector[1] = step * unitVector[1];
		stepVector[2] = step * unitVector[2];

		// Start sampling from the centroid outwards
		double nextPos[3];
		nextPos[0] = centroid[0] + (minRadius - step) * unitVector[0];
		nextPos[1] = centroid[1] + (minRadius - step) * unitVector[1];
		nextPos[2] = centroid[2] + (minRadius - step) * unitVector[2];

		// Keep track of distance
		double dist = 0.0;
		int voxelIdx = -1;

		// Start sampling along ray
		for( int k = 0; k < 500; ++k )
		{
			// Compute voxel space position by dividing by spacing
			double x = nextPos[0] / sx;
			double y = nextPos[1] / sy;
			double z = nextPos[2] / sz;

			// Get interpolated value at given position
			float value = this->getInterpolatedValue<T>( voxels, nx, ny, nz, x, y, z );

			// If value drops below given threshold we passed the tumor boundary.
			// Now iterate back and forth a few times to refine our position
			if( value < threshold )
			{
				double tmpStepVector[] = {
					stepVector[0], stepVector[1], stepVector[2]};

				// Refine sample position
				for( int n = 0; n < 10; ++ n )
				{
					// Divide step vector by half
					tmpStepVector[0] *= 0.5f;
					tmpStepVector[1] *= 0.5f;
					tmpStepVector[2] *= 0.5f;

					// If value is below threshold we outside the tumor so
					// we need to jump back
					if( value < threshold )
					{
						nextPos[0] -= tmpStepVector[0];
						nextPos[1] -= tmpStepVector[1];
						nextPos[2] -= tmpStepVector[2];
					}
					else
					{
						nextPos[0] += tmpStepVector[0];
						nextPos[1] += tmpStepVector[1];
						nextPos[2] += tmpStepVector[2];
					}

					// Lookup new voxel value (after scaling back to
					// voxel space)
					x = nextPos[0] / sx;
					y = nextPos[1] / sy;
					z = nextPos[2] / sz;
					value = this->getInterpolatedValue<T>( voxels, nx, ny, nz, x, y, z );
				}

				// Lookup distance value in distance transform. We use nearest-neighbor
				// interpolation because the distances may not be smooth?
				voxelIdx = this->getNearestVoxelIndex( nx, ny, nz, x, y, z );

				double posA[3];
				posA[0] = sx * x;
				posA[1] = sy * y;
				posA[2] = sz * z;

				double posB[3];
				posB[0] = sx * GET_X( voronoi[voxelIdx] );
				posB[1] = sy * GET_Y( voronoi[voxelIdx] );
				posB[2] = sz * GET_Z( voronoi[voxelIdx] );

				//dist = (double) transform[voxelIdx];

				dist = sqrtf(
					(posB[0] - posA[0]) * (posB[0] - posA[0]) +
					(posB[1] - posA[1]) * (posB[1] - posA[1]) +
					(posB[2] - posA[2]) * (posB[2] - posA[2]) );

				break;
			}

			// Value is still above threshold so update sampling position
			nextPos[0] += stepVector[0];
			nextPos[1] += stepVector[1];
			nextPos[2] += stepVector[2];
		}

		// Return results
		distance = dist;
		voxelIndex = voxelIdx;
	}

	///////////////////////////////////////////////////////////////////////////
	template< class T >
	void DistanceUncertaintyPlugin::computeDistanceMap( T * voxels, float * transform, int * voronoi, double *& map, int *& idxMap,
		int mapRows, int mapCols, double centroid[3], double minRadius, double threshold, int nx, int ny, int nz,
        double sx, double sy, double sz )
	{
		for( int i = 0; i < mapRows * mapCols; ++i )
		{
			map[i] = 99999999.0;
			idxMap[i] = 0;
		}

        double X = 0.0;
        double Y = 0.0;
        double deltaX = 2 * PI / (mapCols - 1);
        double deltaY = PI / (mapRows - 1);

        for( int i = 0; i < mapRows; ++i )
        {
            X = 0.0;

            for( int j = 0; j < mapCols; ++j )
            {
                int voxelIdx    = -1;
                double distance = -1;
                double phi      = Y;
                double theta    = X;

                this->computeDistanceFromSphericalCoordinates( voxels, transform, voronoi, centroid, minRadius,
                    threshold, nx, ny, nz, sx, sy, sz, theta, phi, distance, voxelIdx );

                int x   = (int) ROUND( 255 * X / (2 * PI));
                int y   = (int) ROUND( 127 * Y / PI);
                int idx = y * mapCols + x;

                map[idx] = distance;
                idxMap[idx] = voxelIdx;

                X += deltaX;
            }

            Y += deltaY;
        }
	}

	///////////////////////////////////////////////////////////////////////////
	template< class T >
	void DistanceUncertaintyPlugin::computeSinusoidalDistanceMap( T * voxels, float * transform, int * voronoi, double *& map, int *& idxMap,
		int mapRows, int mapCols, double centroid[3], double minRadius, double threshold, int nx, int ny, int nz,
        double sx, double sy, double sz, double deltaTheta, double deltaPhi )
	{
		for( int i = 0; i < mapRows * mapCols; ++i )
		{
			map[i] = 99999999.0;
			idxMap[i] = 0;
		}

        double X = -PI;
        double Y = -PI / 2;
        double deltaX = 2 * PI / (mapCols - 1);
        double deltaY = PI / (mapRows -1);

        for( int i = 0; i < mapRows; ++i )
        {
            X = -PI;

            for( int j = 0; j < mapCols; ++j )
            {
                int voxelIdx    = -1;
                double distance = -1;
                double phi      = Y;
                double theta    = X / cos( Y );

                this->computeDistanceFromSphericalCoordinates(
                            voxels, transform, voronoi, centroid, minRadius, threshold,
                            nx, ny, nz, sx, sy, sz, theta + deltaTheta, phi + deltaPhi, distance, voxelIdx );

                int x   = (int) ROUND( 255 * (X + PI) / (2 * PI));
                int y   = (int) ROUND( 127 * (Y + PI / 2) / PI);
                int idx = y * mapCols + x;

                if( theta < -PI || theta > PI )
                    map[idx] = 99999999.0;
                else
                    map[idx] = distance;
                idxMap[idx] = voxelIdx;

                X += deltaX;
            }

            Y += deltaY;
        }
	}

	///////////////////////////////////////////////////////////////////////////
	template< class T >
	double DistanceUncertaintyPlugin::getInterpolatedValue( T * voxels, int nx, int ny, int nz,
		double x, double y, double z )
	{
		// See: http://paulbourke.net/miscellaneous/interpolation/index.html

		// Get nearest lower voxel position
		int i = y < 0.0 ? 0 : (int) floor( y );
		int j = x < 0.0 ? 0 : (int) floor( x );
		int k = z < 0.0 ? 0 : (int) floor( z );

		// Translate the interpolation coordinate to the unit cube
		// at the origin of the volume
		double xOrg = x - j;
		double yOrg = y - i;
		double zOrg = z - k;

		// Get voxel values at grid points surrounding our interpolation coordinate
		double v000 = (double) voxels[(k)  *nx*ny+(i)  *nx+(j)  ];
		double v100 = (double) voxels[(k)  *nx*ny+(i)  *nx+(j+1)];
		double v010 = (double) voxels[(k)  *nx*ny+(i+1)*nx+(j)  ];
		double v001 = (double) voxels[(k+1)*nx*ny+(i)  *nx+(j)  ];
		double v101 = (double) voxels[(k+1)*nx*ny+(i)  *nx+(j+1)];
		double v011 = (double) voxels[(k+1)*nx*ny+(i+1)*nx+(j)  ];
		double v110 = (double) voxels[(k)  *nx*ny+(i+1)*nx+(j+1)];
		double v111 = (double) voxels[(k+1)*nx*ny+(i+1)*nx+(j+1)];

		// Compute interpolated value
		double value =
				v000 * (1 - xOrg) * (1 - yOrg) * (1 - zOrg) +
				v100 * xOrg * (1 - yOrg) * (1 - zOrg) +
				v010 * (1 - xOrg) * yOrg * (1 - zOrg) +
				v001 * (1 - xOrg) * (1 - yOrg) * zOrg +
				v101 * xOrg * (1 - yOrg) * zOrg +
				v011 * (1 - xOrg) * yOrg * zOrg +
				v110 * xOrg * yOrg * (1 - zOrg) +
				v111 * xOrg * yOrg * zOrg;

		return value;
	}

	///////////////////////////////////////////////////////////////////////////
	int DistanceUncertaintyPlugin::getNearestVoxelIndex( int nx, int ny, int nz, double x, double y, double z )
	{
		// Get nearest lower voxel position
		int i = y < 0.0 ? 0 : (int) floor( y );
		int j = x < 0.0 ? 0 : (int) floor( x );
		int k = z < 0.0 ? 0 : (int) floor( z );

		// Compute distances between (x,y,z) and 8 corner points
		double distance[8];
		distance[0] = sqrt( (x-j)*(x-j) + (y-i)*(y-i) + (z-k)*(z-k) );
		distance[1] = sqrt( (x-(j+1))*(x-(j+1)) + (y-i)*(y-i) + (z-k)*(z-k) );
		distance[2] = sqrt( (x-j)*(x-j) + (y-(i+1))*(y-(i+1)) + (z-k)*(z-k) );
		distance[3] = sqrt( (x-j)*(x-j) + (y-i)*(y-i) + (z-(k+1))*(z-(k+1)) );
		distance[4] = sqrt( (x-(j+1))*(x-(j+1)) + (y-i)*(y-i) + (z-(k+1))*(z-(k+1)) );
		distance[5] = sqrt( (x-j)*(x-j) + (y-(i+1))*(y-(i+1)) + (z-(k+1))*(z-(k+1)) );
		distance[6] = sqrt( (x-(j+1))*(x-(j+1)) + (y-(i+1))*(y-(i+1)) + (z-k)*(z-k) );
		distance[7] = sqrt( (x-(j+1))*(x-(j+1)) + (y-(i+1))*(y-(i+1)) + (z-(k+1))*(z-(k+1)) );

		// Compute voxel positions of 8 corner points
		int indexes[8];
		indexes[0] = (k)  *nx*ny+(i)  *nx+(j);
		indexes[1] = (k)  *nx*ny+(i)  *nx+(j+1);
		indexes[2] = (k)  *nx*ny+(i+1)*nx+(j);
		indexes[3] = (k+1)*nx*ny+(i)  *nx+(j);
		indexes[4] = (k+1)*nx*ny+(i)  *nx+(j+1);
		indexes[5] = (k+1)*nx*ny+(i+1)*nx+(j);
		indexes[6] = (k)  *nx*ny+(i+1)*nx+(j+1);
		indexes[7] = (k+1)*nx*ny+(i+1)*nx+(j+1);

		// Find voxel closest to coordinate (x,y,z)
		int idx = -1;
		double min = 9999999.0;
		for( int i = 0; i < 8; ++i )
		{
			if( distance[i] < min )
			{
				min = distance[i];
				idx = i;
			}
		}

		return indexes[idx];
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::computeMinMaxDistancePosition( int ** voxelPositions, int nrPositions,
		float * transformVoxels, int * voronoiVoxels, double & minDist, double minDistPos1[3],
		double minDistPos2[3], double & maxDist, int nx, int ny, int nz, double sx, double sy, double sz )
	{
		double min = 99999999.0;
		double max = 0.0;
		double P1[3], P2[3];

		for( int i = 0; i < nrPositions; ++i )
		{
			int x = voxelPositions[i][0];
			int y = voxelPositions[i][1];
			int z = voxelPositions[i][2];

			int idx = z * nx * ny + y * nx + x;

			int value = voronoiVoxels[idx];

			double posA[3];
			posA[0] = sx * x;
			posA[1] = sy * y;
			posA[2] = sz * z;

			double posB[3];
			posB[0] = sx * GET_X( value );
			posB[1] = sy * GET_Y( value );
			posB[2] = sz * GET_Z( value );

			double distance = sqrtf(
				(posB[0] - posA[0]) * (posB[0] - posA[0]) +
				(posB[1] - posA[1]) * (posB[1] - posA[1]) +
				(posB[2] - posA[2]) * (posB[2] - posA[2]) );

			if( distance < min )
			{
				min = distance;

				for( int i = 0; i < 3; ++i )
				{
					P1[i] = posA[i];
					P2[i] = posB[i];
				}
			}

			if( distance > max )
				max = distance;
		}

		minDistPos1[0] = P1[0];
		minDistPos1[1] = P1[1];
		minDistPos1[2] = P1[2];

		minDistPos2[0] = P2[0];
		minDistPos2[1] = P2[1];
		minDistPos2[2] = P2[2];

		minDist = min;
		maxDist = max;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::normalizeVector( double V[3] )
	{
		double length = sqrt( V[0]*V[0] + V[1]*V[1] + V[2]*V[2] );
		V[0] /= length;
		V[1] /= length;
		V[2] /= length;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::sphericalToCartesian( double R, double theta, double phi, double V[3] )
	{
		V[0] = R * cos( theta ) * sin( phi );
		V[1] = R * sin( theta ) * sin( phi );
		V[2] = R * cos( phi );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::cartesianToSpherical( double V[3], double & R, double & theta, double & phi )
	{
		R = sqrt( V[0]*V[0] + V[1]*V[1] + V[2]*V[2] );
		phi = acos( V[2] / R );
		theta = (V[0] == 0) ? 0.0 : atan2( V[1], V[0] );
	}


	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::clearArrowWidgets()
	{
		vtkRenderer * renderer = this->fullCore()->canvas()->GetRenderer3D();
		if( renderer == 0 )
			return;

		for( int i = 0; i < _arrowWidgets.size(); ++i )
		{
			vtkDistanceArrowWidget * widget = _arrowWidgets.at( i );
			if( renderer->HasViewProp( widget ) )
				renderer->RemoveViewProp( widget );
			widget->Delete();
			widget = 0;
		}

		_arrowWidgets.clear();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::addArrowWidget( vtkDistanceArrowWidget * widget )
	{
		vtkRenderer * renderer = this->fullCore()->canvas()->GetRenderer3D();
		if( renderer == 0 )
			return;

		_arrowWidgets.append( widget );

		this->fullCore()->canvas()->GetRenderer3D()->AddViewProp( widget );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::addArrowWidget( double P[3], double Q[3], double distance )
	{
		vtkDistanceArrowWidget * widget = vtkDistanceArrowWidget::New();
		widget->SetStartPoint( P );
		widget->SetEndPoint( Q );
		widget->SetDistance( distance );
		widget->UpdateGeometry();

		this->addArrowWidget( widget );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::clearPointMarkerWidgets()
	{
		vtkRenderer * renderer = this->fullCore()->canvas()->GetRenderer3D();
		if( renderer == 0 )
			return;

		for( int i = 0; i < _pointMarkerWidgets.size(); ++i )
		{
			vtkPointMarkerWidget * widget = _pointMarkerWidgets.at( i );
			if( renderer->HasViewProp( widget ) )
				renderer->RemoveViewProp( widget );
			widget->Delete();
			widget = 0;
		}

		_pointMarkerWidgets.clear();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::addPointMarkerWidget( vtkPointMarkerWidget * widget )
	{
		vtkRenderer * renderer = this->fullCore()->canvas()->GetRenderer3D();
		if( renderer == 0 )
			return;

		_pointMarkerWidgets.append( widget );

		this->fullCore()->canvas()->GetRenderer3D()->AddViewProp( widget );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::addPointMarkerWidget( double P[3] )
	{
		vtkPointMarkerWidget * widget = vtkPointMarkerWidget::New();
		widget->SetPosition( P );
		widget->SetSize( 2 );
		widget->SetColor( 0, 0, 1 );
		widget->UpdateGeometry();

		this->addPointMarkerWidget( widget );
	}

    ///////////////////////////////////////////////////////////////////////////
    vtkTexture * DistanceUncertaintyPlugin::buildTexture( double * dataA, double * dataB, int rows, int columns, double threshold )
    {
		return NULL;
    }

    ///////////////////////////////////////////////////////////////////////////
	vtkTexture * DistanceUncertaintyPlugin::buildTexture( double * data, int rows, int columns, double threshold )
	{
		vtkImageData * imageData = vtkImageData::New();
		imageData->SetScalarTypeToDouble();
		imageData->SetNumberOfScalarComponents( 1 );
		imageData->SetSpacing( 1, 1, 0 );
		imageData->SetExtent( 0, columns - 1, 0, rows - 1, 0, 0 );
		imageData->AllocateScalars();

		double * pointer = (double *) imageData->GetScalarPointer();
		for( int i = 0; i < rows * columns; ++i )
			pointer[i] = data[i];

		double * range = imageData->GetScalarRange();
		int thresholdIdx = (int) 255 * (threshold - range[0]) / (range[1] - range[0]);

		vtkLookupTable * lut = vtkLookupTable::New();
		lut->Allocate( 256 );
		lut->SetTableRange( range );

		for( int i = 0; i < 256; ++i )
		{
            lut->SetTableValue( i, i / 255.0, i / 255.0, i / 255.0 );
		}

        lut->Build();

		vtkTexture * texture = vtkTexture::New();
		texture->SetInput( imageData );
		texture->SetBlendingMode( vtkTexture::VTK_TEXTURE_BLENDING_MODE_MODULATE );
		texture->InterpolateOn();
		texture->RepeatOn();
		texture->SetLookupTable( lut );

		imageData->Delete();
		lut->Delete();

		return texture;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::updateViewPoint( double P[3] )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );

		vtkCamera * camera = this->getRenderer3D()->GetActiveCamera();

		// Get camera position
		double camPos[3];
		camera->GetPosition( camPos );

		// Get tumor centroid
		double tumorCentroid[3];
		tumorCentroid[0] = _currentConfig->getTumorCentroidX();
		tumorCentroid[1] = _currentConfig->getTumorCentroidY();
		tumorCentroid[2] = _currentConfig->getTumorCentroidZ();

		// Compute unit vector from tumor centroid to camera position
		double U[3], Ulength;
		vtkMath::Subtract( camPos, tumorCentroid, U );
		Ulength = vtkMath::Norm( U );
		vtkMath::Normalize( U );

		// Compute unit vector from tumor centroid to P
		double V[3], Vlength;
		vtkMath::Subtract( P, tumorCentroid, V );
		Vlength = vtkMath::Norm( V );
		vtkMath::Normalize( V );

		// Compute unit vector perpendicular to U and V
		double W[3];
		vtkMath::Cross( U, V, W );

		// Compute angle between U and V
		int nrSteps = _distanceWidget->cameraRotationSpeed();
		double angleEnd = acos( vtkMath::Dot( U, V ) );
		double angleStep = angleEnd / nrSteps;
		double angleStart = 0.0;

		// Rotate towards the selected point in animated fashion
		for( int i = 0; i < nrSteps; ++i )
		{
			double Q[4];
			Q[0] = W[0] * sin( 0.5 * angleStart );
			Q[1] = W[1] * sin( 0.5 * angleStart );
			Q[2] = W[2] * sin( 0.5 * angleStart );
			Q[3] = cos( 0.5 * angleStart );

			double Qconj[4];
			this->quaternionConjugate( Q, Qconj );

			double QU[4] = { U[0], U[1], U[2], 0.0 };
			double tmp[4], Unew[4];
			this->multiplyQuaternions( QU, Qconj, tmp );
			this->multiplyQuaternions( Q, tmp, Unew );

			double Unorm[3];
			Unorm[0] = Unew[0];
			Unorm[1] = Unew[1];
			Unorm[2] = Unew[2];
			vtkMath::Normalize( Unorm );

			double newPos[3];
			newPos[0] = Ulength * Unorm[0] + tumorCentroid[0];
			newPos[1] = Ulength * Unorm[1] + tumorCentroid[1];
			newPos[2] = Ulength * Unorm[2] + tumorCentroid[2];
			camera->SetPosition( newPos );
			camera->SetFocalPoint( tumorCentroid );

			this->fullCore()->render();
			this->getRenderer3D()->ResetCameraClippingRange();

			if( angleEnd >= 0 )
				angleStart += angleStep;
			else
				angleStart -= angleStep;
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::multiplyQuaternions( double P[4], double Q[4], double R[4] )
	{
		double Px = P[0]; double Py = P[1]; double Pz = P[2]; double Pw = P[3];
		double Qx = Q[0]; double Qy = Q[1]; double Qz = Q[2]; double Qw = Q[3];

		R[0] = Pw * Qx + Px * Qw + Py * Qz - Pz * Qy;
		R[1] = Pw * Qy + Py * Qw + Pz * Qx - Px * Qz;
		R[2] = Pw * Qz + Pz * Qw + Px * Qy - Py * Qx;
		R[3] = Pw * Qw - Px * Qx - Py * Qy - Pz * Qz;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::quaternionConjugate( double P[4], double Q[4]  )
	{
		Q[0] = -P[0];
		Q[1] = -P[1];
		Q[2] = -P[2];
		Q[3] =  P[3];
	}

	///////////////////////////////////////////////////////////////////////////
	vtkActor * DistanceUncertaintyPlugin::findActor( vtkPolyData * polyData )
	{
		plugin::Manager * manager = this->fullCore()->plugin();
		PolyDataVisualizationPlugin * plugin =
				(PolyDataVisualizationPlugin *) manager->getPlugin( manager->indexOf( "PolyData" ) );

		vtkPropAssembly * assembly = (vtkPropAssembly * ) plugin->getVtkProp();
		vtkPropCollection * props = assembly->GetParts();
		vtkActor * prop = NULL;

		props->InitTraversal();
		while( prop = (vtkActor *) props->GetNextProp() )
		{
			vtkPolyDataMapper * mapper = (vtkPolyDataMapper *) prop->GetMapper();
			if( polyData == mapper->GetInput() )
			{
				return prop;
			}
		}

		return NULL;
	}

	///////////////////////////////////////////////////////////////////////////
	RayCastVolumeMapper * DistanceUncertaintyPlugin::findVolumeMapper()
	{
		plugin::Manager * manager = this->fullCore()->plugin();
		RayCastPlugin * plugin = (RayCastPlugin *) manager->getPlugin( manager->indexOf( "Ray Cast Plugin" ) );
		vtkVolume * volume = (vtkVolume *) plugin->getVtkProp();
		RayCastVolumeMapper * volumeMapper = (RayCastVolumeMapper *) volume->GetMapper();
		return volumeMapper;
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceConfiguration * DistanceUncertaintyPlugin::findDistanceConfiguration( QString name )
	{
		for( int i = 0; i < _configurations.size(); ++i )
		{
			DistanceConfiguration * config = _configurations.at( i );
			if( config->getName() == name )
				return config;
		}

		return NULL;
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceConfiguration * DistanceUncertaintyPlugin::findDistanceConfiguration( vtkImageData * tumorData, vtkImageData * fiberData )
	{
		for( int i = 0; i < _configurations.size(); ++i )
		{
			DistanceConfiguration * config = _configurations.at( i );
			if( tumorData == config->getTumorData() && fiberData == config->getFiberData() )
				return config;
		}

		return NULL;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::showPickPosition( double position[3] )
	{
		if( _currentConfig == 0 )
			return;
		Q_ASSERT( _currentConfig->isValid() );
		Q_ASSERT( _currentConfig->getSelectedVoronoiData() );

		double tumorCentroid[3];
		tumorCentroid[0] = _currentConfig->getTumorCentroidX();
		tumorCentroid[1] = _currentConfig->getTumorCentroidY();
		tumorCentroid[2] = _currentConfig->getTumorCentroidZ();

		double V[3];
		V[0] = position[0] - tumorCentroid[0];
		V[1] = position[1] - tumorCentroid[1];
		V[2] = position[2] - tumorCentroid[2];

		// Translate 3D point to (theta,phi) pair on the sphere
		double R, theta, phi;
		this->cartesianToSpherical( V, R, theta, phi );

//		// Translate (theta,phi) pair to row and column in map depending on projection
//		int y = (int) round( 127 * (phi + PI/2) / PI );
//		int x = (int) round( 255 * (theta * cos( phi ) + PI ) / (2*PI) );

//		vtkImageData * voronoi = _currentConfig->getSelectedVoronoiData();
//		int * voronoiVoxels = (int *) voronoi->GetScalarPointer();
//		int voxelIdx = this->getNearestVoxelIndex(
//					_currentConfig->getDimensionX(),
//					_currentConfig->getDimensionY(),
//					_currentConfig->getDimensionZ(), position[0], position[1], position[2] );
//		_distanceWidget->map()->canvas()->setSelectedVoxelIndex( voxelIdx );
	}

	// CELL PICKER EVENT HANDLER

	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin::CellPickerEventHandler::CellPickerEventHandler( DistanceUncertaintyPlugin * plugin, vtkInteractorStyleCellPicker * picker )
	{
		this->Plugin = plugin;
		this->Picker = picker;
		this->Picker->SetEventHandler( this );
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::CellPickerEventHandler::Execute( vtkObject * caller, unsigned long eventId, void * callData )
	{
		double position[3];
		this->Picker->GetPickedPosition( position );
		std::cout << "CellPickerEventHandler::Execute() position = " <<
					 position[0] << " " <<
					 position[1] << " " << position[2] << std::endl;
		this->Plugin->showPickPosition( position );
	}
}

Q_EXPORT_PLUGIN2( libDistanceUncertaintyPlugin, bmia::DistanceUncertaintyPlugin )
