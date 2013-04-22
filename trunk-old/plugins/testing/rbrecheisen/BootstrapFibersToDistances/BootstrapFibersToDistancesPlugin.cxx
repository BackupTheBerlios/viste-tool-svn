// Includes DTITool
#include <BootstrapFibersToDistancesPlugin.h>
#include <vtkBootstrapStreamlineToDistanceTableFilter.h>
#include <vtkDistanceTable.h>
#include <vtkDistanceMeasure.h>
#include <vtkDistanceMeasureClosestPointDistance.h>
#include <vtkDistanceMeasureEndPointDistance.h>
#include <vtkDistanceMeasureMeanOfClosestPointDistances.h>

// Includes QT
#include <QFileDialog>

// Includes VTK
#include <vtkPolyDataReader.h>
#include <vtkArrayReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkArrayWriter.h>

// Include C++
#include <fstream>

// Definition of cache directory. This directory is used to
// store intermediate computation results
#define CACHE_DIRECTORY "/Users/Ralph/Temp/Cache/"

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	BootstrapFibersToDistancesPlugin::BootstrapFibersToDistancesPlugin() :
		plugin::Plugin( "BootstrapFibersToDistancesPlugin" ),
		plugin::GUI(),
		data::Consumer()
	{
		// Create QT widget that will hold plugin's GUI
		_widget = new QWidget;

		// Create UI form that describes plugin's GUI, then
		// initialize widget with the form
		_form = new Ui::BootstrapFibersToDistancesForm;
		_form->setupUi( _widget );

		// Setup distance measures
		this->setupDistanceMeasures();

		// Setup signal/slot connections
		this->setupConnections();
	}

	///////////////////////////////////////////////////////////////////////////
	BootstrapFibersToDistancesPlugin::~BootstrapFibersToDistancesPlugin()
	{
		// Delete QT objects
		delete _widget;
		delete _form;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * BootstrapFibersToDistancesPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::dataSetAdded( data::DataSet * dataset )
	{
		// Return if there's nothing there
		if( ! dataset )
			return;

		// We only handle bootstrap fibers or bootstrap fiber ID's
		if( dataset->getKind() == "bootstrapFibers" )
		{
			// Check that we have polydata
			vtkPolyData * fibers = dataset->getVtkPolyData();
			if( ! fibers )
			{
				std::cout << "BootstrapFibersToDistancesPlugin::dataSetAdded() ";
				std::cout << "dataset is not vtkPolyData!" << std::endl;
				return;
			}

			// Add fibers to the list
			_fiberSets.append( fibers );

			// Add name of the dataset to combo box
			_form->comboFibers->addItem( dataset->getName() );
		}
		else if( dataset->getKind() == "bootstrapFiberIds" )
		{
			// Check that we have a VTK int array
			vtkIntArray * fiberIds = vtkIntArray::SafeDownCast( dataset->getVtkObject() );
			if( ! fiberIds )
			{
				std::cout << "BootstrapFibersToDistancesPlugin::dataSetAdded() ";
				std::cout << "dataset is not vtkIntArray!" << std::endl;
				return;
			}

			// Get number of seed points from the attribtues
			int numberOfPoints = 0;
			if( ! dataset->getAttributes()->getAttribute( "numberOfSeedPoints", numberOfPoints ) )
			{
				std::cout << "BootstrapFibersToDistancesPlugin::dataSetAdded() ";
				std::cout << "dataset does not contain 'numberOfSeedPoints' attribute!" << std::endl;
				return;
			}

			// Add array to the list
			_fiberIdSets.append( fiberIds );

			// Add number of seed points to the list
			_numberSeedPointsList.append( numberOfPoints );

			// Add name to the combo box
			_form->comboFiberIds->addItem( dataset->getName() );
		}
		else {}
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::compute()
	{
		// Get selected fibers from list
		int fiberIndex = _form->comboFibers->currentIndex();
		vtkPolyData * fibers = _fiberSets.at( fiberIndex );

		// Get selected fiber ID's from the list
		int idIndex = _form->comboFiberIds->currentIndex();
		vtkIntArray * fiberIds = _fiberIdSets.at( idIndex );

		// Get number of seed points from the list
		int nrPoints = _numberSeedPointsList.at( idIndex );

		// Create distance measure object based on user selection
		QString tableName = _form->comboFibers->currentText();
		vtkDistanceMeasure * measure = 0;
		switch( _form->comboDistanceMeasure->currentIndex() )
		{
		case 0:
			measure = vtkDistanceMeasureClosestPointDistance::New();
			tableName.append( "ClosestPoint" );
			break;
		case 1:
			measure = vtkDistanceMeasureEndPointDistance::New();
			tableName.append( "EndPoint" );
			break;
		case 2:
			measure = vtkDistanceMeasureMeanOfClosestPointDistances::New();
			tableName.append( "MeanOfClosestPoints" );
			break;
		default:
			std::cout << "BootstrapFibersToDistancesPlugin::compute() unknown measure" << std::endl;
		}

		// Create distance filter
		vtkBootstrapStreamlineToDistanceTableFilter * filter = vtkBootstrapStreamlineToDistanceTableFilter::New();
		filter->SetInput( fibers );
		filter->SetFiberIds( fiberIds );
		filter->SetNumberOfSeedPoints( nrPoints );
		filter->SetDistanceMeasure( measure );

		// Execute the distance filter. This may take a while depending
		// on which distance measure was selected
		filter->Execute();

		// Add table to the map. The name will be a combination of the
		// fiber name and the distance measure used
		vtkDistanceTable * table = filter->GetOutput();
		_tableSets[ tableName ] = table;

		// Add name of distance table to combo box
		_form->comboDistanceTable->addItem( tableName );
		_form->comboDistanceTable->setCurrentIndex( _form->comboDistanceTable->count() - 1 );

		std::cout << "BootstrapFibersToDistancesPlugin::compute() ";
		std::cout << "inserted distance table at index " << fiberIndex << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::load()
	{
		// Show file dialog that allows user to select fibers. This will
		// automatically also load the associated fiber ID's (which have
		// the same filename base)
		QString fileNameAndPath = QFileDialog::getOpenFileName(
					0, "Select Fibers", CACHE_DIRECTORY, "*.fbs" );
		if( fileNameAndPath.isEmpty() )
			return;

		// Create base filename without extension
		QString baseName = fileNameAndPath;
		baseName.remove( baseName.length() - 4, 4 );

		// Create filename without path
		QString fileName = baseName;
		int index = fileName.lastIndexOf( '/' );
		if( index < 0 )
			index = fileName.lastIndexOf( '\\' );
		fileName.remove( 0, index + 1 );

		// Load fibers from file. For now, the fibers are not added to
		// the data manager. We'll have to see what to do about that
		// in the future. It's possible to load the fibers separately
		// with the File->Open menu in DTI tool but then it will be
		// loaded twice...
		vtkPolyDataReader * reader = vtkPolyDataReader::New();
		reader->SetFileName( fileNameAndPath.toStdString().c_str() );
		reader->Update();
		vtkPolyData * fibers = reader->GetOutput();
		fibers->Register( 0 );
		data::DataSet * datasetFibers = new data::DataSet( fileName, "bootstrapFibers", fibers );
		this->core()->data()->addDataSet( datasetFibers );
		reader->Delete();

		// Load fiber ID's
		std::ifstream idsFile;
		QString idsFileName = baseName + "_ids.txt";
		idsFile.open( idsFileName.toStdString().c_str() );
		if( idsFile.bad() )
		{
			std::cout << "BootstrapFibersToDistancesPlugin::load() ";
			std::cout << "could not open ID file for reading" << std::endl;
			return;
		}

		int nrIds;
		idsFile >> nrIds;
		vtkIntArray * ids = vtkIntArray::New();
		ids->Allocate( nrIds );
		for( int i = 0; i < nrIds; ++i )
		{
			int id;
			idsFile >> id;
			ids->InsertNextValue( id );
		}
		idsFile.close();

		// Load number of seed points
		std::ifstream nrSeedsFile;
		QString nrSeedsFileName = baseName + "_nrseeds.txt";
		nrSeedsFile.open( nrSeedsFileName.toStdString().c_str() );
		if( nrSeedsFile.bad() )
		{
			std::cout << "BootstrapFibersToDistancesPlugin::load() ";
			std::cout << "could not open nr. seeds file for reading" << std::endl;
			return;
		}

		int nrSeeds;
		nrSeedsFile >> nrSeeds;
		nrSeedsFile.close();

		// Create new dataset for fiber ID's and set number of seed points
		// as attribute of dataset. This will result in the dataSetAdded()
		// method being called
		data::DataSet * datasetIds = new data::DataSet( fileName + "Ids", "bootstrapFiberIds", ids );
		datasetIds->getAttributes()->addAttribute( "numberOfSeedPoints", nrSeeds );
		this->core()->data()->addDataSet( datasetIds );

		// Load distance table (if available). This requires figuring what
		// type of distance measure was used based on the filename
		QFileInfo infoClosestPoint( baseName + "_closestpoint.txt" );
		QFileInfo infoEndPoint( baseName + "_endpoint.txt" );
		QFileInfo infoMeanOfClosestPoints( baseName + "_meanofclosestpoints.txt" );
		QString measureName = "";

		if( infoClosestPoint.exists() )
			measureName = "ClosestPoint";
		else if( infoEndPoint.exists() )
			measureName = "EndPoint";
		else if( infoMeanOfClosestPoints.exists() )
			measureName = "MeanOfClosestPoints";
		else {}

		if( measureName != "" )
		{
			QString tableFileName = baseName + "_" + measureName.toLower() + ".txt";
			vtkDistanceTable * table = vtkDistanceTable::New();
			table->Read( tableFileName.toStdString().c_str() );
			_tableSets[ fileName + measureName ] = table;
			_form->comboDistanceTable->addItem( fileName + measureName );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::save()
	{
		// Show file dialog that allows the user to save the fibers to
		// disk. This will automatically also save the fiber ID's with
		// the same filename base
		QString fileNameAndPath = QFileDialog::getSaveFileName( 0, "Choose File Name", CACHE_DIRECTORY );
		if( fileNameAndPath.isEmpty() )
			return;

		// Get currently selected fibers
		int fiberIndex = _form->comboFibers->currentIndex();
		vtkPolyData * fibers = _fiberSets.at( fiberIndex );
		QString fibersFileName = fileNameAndPath + ".fbs";
		vtkPolyDataWriter * fibersWriter = vtkPolyDataWriter::New();
		fibersWriter->SetFileName( fibersFileName.toStdString().c_str() );
		fibersWriter->SetInput( fibers );
		fibersWriter->Write();
		fibersWriter->Delete();

		// Get currently selected fiber ID's and write them to a text
		// file with same name as fibers but additional '_ids.vtk'
		// appended to it
		int idIndex = _form->comboFiberIds->currentIndex();
		vtkIntArray * fiberIds = _fiberIdSets.at( idIndex );
		QString idsFileName = fileNameAndPath + "_ids.txt";
		std::ofstream idsFile;
		idsFile.open( idsFileName.toStdString().c_str() );
		idsFile << fiberIds->GetNumberOfTuples() << std::endl;
		for( int i = 0; i < fiberIds->GetNumberOfTuples(); ++i )
		{
			int * value = new int[1];
			fiberIds->GetTupleValue( i, value );
			idsFile << value[0] << std::endl;
		}
		idsFile.close();

		// Get number of seed points from the list
		int nrSeeds = _numberSeedPointsList.at( idIndex );
		QString nrSeedsFileName = fileNameAndPath + "_nrseeds.txt";
		std::ofstream nrSeedsFile;
		nrSeedsFile.open( nrSeedsFileName.toStdString().c_str() );
		nrSeedsFile << nrSeeds << std::endl;
		nrSeedsFile.close();

		// Get currently selected distance table (if computed)
		if( _form->comboDistanceTable->count() > 0 )
		{
			QString tableName = _form->comboFibers->currentText();
			QString measureName = "";
			if( _form->comboDistanceMeasure->currentText() == "End Point" )
				measureName = "EndPoint";
			else if( _form->comboDistanceMeasure->currentText() == "Mean of Closest Points" )
				measureName = "MeanOfClosestPoints";
			else
				measureName = "ClosestPoint";
			vtkDistanceTable * table = _tableSets[ tableName + measureName ];
			QString tableFileName = fileNameAndPath + "_" + measureName.toLower() + ".txt";
			table->Write( tableFileName.toStdString() );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::dataSetRemoved( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::dataSetChanged( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::setupDistanceMeasures()
	{
		_form->comboDistanceMeasure->addItem( "Closest Point" );
		_form->comboDistanceMeasure->addItem( "End Point" );
		_form->comboDistanceMeasure->addItem( "Mean of Closest Points" );
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFibersToDistancesPlugin::setupConnections()
	{
		this->connect( _form->buttonCompute, SIGNAL( clicked() ), this, SLOT( compute() ) );
		this->connect( _form->buttonLoad, SIGNAL( clicked() ), this, SLOT( load() ) );
		this->connect( _form->buttonSave, SIGNAL( clicked() ), this, SLOT( save() ) );
	}
}

Q_EXPORT_PLUGIN2( libBootstrapFibersToDistancesPlugin, bmia::BootstrapFibersToDistancesPlugin )

#undef CACHE_DIRECTORY
