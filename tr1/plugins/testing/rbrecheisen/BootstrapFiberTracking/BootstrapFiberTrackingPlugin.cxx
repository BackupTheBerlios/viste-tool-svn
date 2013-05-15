// Includes DTITool
#include <BootstrapFiberTrackingPlugin.h>
#include <AnisotropyMeasures.h>
#include <vtkBootstrapFiberTrackingFilter.h>

// Includes QT
#include <QFileDialog>
#include <QSettings>

// Includes VTK
#include <vtkActor.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkIntArray.h>

// Includes STL
#include <vector>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	BootstrapFiberTrackingPlugin::BootstrapFiberTrackingPlugin() :
		plugin::Plugin( "BootstrapFiberTrackingPlugin" ),
		plugin::Visualization(),
		plugin::GUI(),
		data::Consumer()
	{
		// Create QT widget that will hold plugin's GUI
		_widget = new QWidget;

		// Create UI form that describes plugin's GUI, then
		// initialize widget with the form
		_form = new Ui::BootstrapFiberTrackingForm;
		_form->setupUi( _widget );

		// Create empty VTK actor so we can return something
		// when the plugin manager asks for it
		_prop = vtkActor::New();

		// Setup signal/slot connections
		this->setupConnections();
	}

	///////////////////////////////////////////////////////////////////////////
	BootstrapFiberTrackingPlugin::~BootstrapFiberTrackingPlugin()
	{
		// Delete QT objects
		delete _widget;
		delete _form;

		// Delete VTK objects
		_prop->Delete();
	}

	///////////////////////////////////////////////////////////////////////////
	vtkProp * BootstrapFiberTrackingPlugin::getVtkProp()
	{
		return _prop;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * BootstrapFiberTrackingPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::dataSetAdded( data::DataSet * dataset )
	{
		// Make sure we have a non-null dataset
		if( dataset == 0 )
			return;

		if( dataset->getKind() == "DTI" )
		{
			// Check that we have valid image data
			vtkImageData * image = dataset->getVtkImageData();
			Q_ASSERT( image );

			// Add name of dataset to DTI volume list
			_tensorVolumes.append( image );

			// Also add it to the combo box
			_form->comboDatasets->addItem( dataset->getName() );
		}
		else if( dataset->getKind() == "scalar volume" )
		{
			// Check that we have valid image data
			vtkImageData * image = dataset->getVtkImageData();
			Q_ASSERT( image );

			// Add name of dataset to anisotropy volume list
			_aiVolumes.append( image );

			// Add name to combo box
			_form->comboAnisotropyDatasets->addItem( dataset->getName() );
		}
		else if( dataset->getKind() == "seed points" )
		{
			// Add ROI's name to combo box
			QString name = dataset->getName();
			_form->comboSeedRegions->addItem( name );

			// Store the associated dataset in our list
			vtkUnstructuredGrid * seeds =
					vtkUnstructuredGrid::SafeDownCast( dataset->getVtkObject() );
			Q_ASSERT( seeds );
			_seedRegions.append( seeds );
		}
		else {}
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::dataSetRemoved( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::dataSetChanged( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::openDir()
	{
		// Popup file dialog to let user select first bootstrap
		// file in the directory
		QString fileNameAndPath = QFileDialog::getOpenFileName( 0, "Select First Bootstrap File", "." );
		if( fileNameAndPath.isEmpty() )
			return;

		// Get directory associated with file name and directory
		QString directory = this->getDirectory( fileNameAndPath );
		if( directory.isEmpty() )
			return;

		// Get file name (without path)
		QString fileName = this->getFileName( fileNameAndPath );

		// Get a list of bootstrap filenames. We use this to load
		// each tensor volume and perform fiber tracking on it
		if( _fileNames.count() > 0 )
			_fileNames.clear();
		_fileNames = this->getBootstrapFileNames( directory, fileName );
		if( _fileNames.count() == 0 )
			return;

		// Update bootstraps found label
		QString text = QString( "%1" ).arg( _fileNames.count() );
		_form->labelNrBootstrapsFound->setText( text );
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::run()
	{
		// Check that the user did not choose more iterations than we
		// have bootstrap files. If he did, then just use the number
		// of bootstrap files
		int numberOfIterations = _form->sliderNrBootstraps->value();
		if( numberOfIterations > _fileNames.count() )
			numberOfIterations = _fileNames.count();

		// Get anistropy index image
		int index = _form->comboAnisotropyDatasets->currentIndex();
		vtkImageData * anisotropy = _aiVolumes.at( index );

		// Get seed points
		index = _form->comboSeedRegions->currentIndex();
		vtkUnstructuredGrid * seedPoints = _seedRegions.at( index );

		// Create bootstrap fiber tracker
		vtkBootstrapFiberTrackingFilter * filter = vtkBootstrapFiberTrackingFilter::New();

		// Set parameters
		filter->SetMaximumPropagationDistance( 1000.0f );
		filter->SetIntegrationStepLength( 0.1f );
		filter->SetMinimumFiberSize( 0.0f );
		filter->SetSimplificationStepLength( 1.0f );
		filter->SetStopAIValue( _form->sliderAnisotropy->value() / 100.0f );
		filter->SetStopDegrees( 90.0f );
		filter->SetSeedPoints( seedPoints );
		filter->SetAnisotropyIndexImage( anisotropy );

		// Set file names
		std::vector< std::string > fileNames;
		for( int i = 0; i < numberOfIterations; ++i )
			fileNames.push_back( _fileNames.at( i ).toStdString() );
		filter->SetFileNames( fileNames );

		// Run bootstrap tracker
		filter->Update();

		// Get output and fiber ID's
		vtkPolyData * fibers = filter->GetOutput();
		vtkIntArray * fiberIds = filter->GetFiberIds();

		// Create dataset and add bootstrap fibers to data repository
		data::DataSet * fiberDataset = new data::DataSet( "bootstrapFibers", "bootstrapFibers", fibers );
		this->core()->data()->addDataSet( fiberDataset );

		// Create dataset for the fiber ID's. Also add an attribute that
		// specifies the number of seed points
		data::DataSet * idDataset = new data::DataSet( "bootstrapFiberIds", "bootstrapFiberIds", fiberIds );
		int numberOfPoints = seedPoints->GetNumberOfPoints();
		idDataset->getAttributes()->addAttribute( "numberOfSeedPoints", numberOfPoints );

		// Add fiber ID's to data repository so that the distance
		// computation can get at it
		this->core()->data()->addDataSet( idDataset );

		// Delete filter. This should not affect the fibers
		// because they are only unregistered in the bootstrap
		// tracker class. The same holds for fiber ID's
		filter->Delete();
	}

	///////////////////////////////////////////////////////////////////////////
	QString BootstrapFiberTrackingPlugin::getDirectory( const QString fileName )
	{
		QString directory;
		QString str = fileName;

		// Find index of last forward or backward slash
		int index = fileName.lastIndexOf( '/' );
		if( index < 0 )
			index = fileName.lastIndexOf( '\\' );

		// If the index is negative, this means there's not directory
		// information in the file name. Just return without any
		// errors because it's not uncommon
		if( index < 0 )
			return directory;

		// Cut file name at the slash and return it
		str.truncate( index );
		directory = str;
		return directory;
	}

	///////////////////////////////////////////////////////////////////////////
	QString BootstrapFiberTrackingPlugin::getFileName( const QString fileNameAndPath )
	{
		// Find index of last forward or backward slash
		int index = fileNameAndPath.lastIndexOf( '/' );
		if( index < 0 )
			index = fileNameAndPath.lastIndexOf( '\\' );

		// If we could not find slashes, we're assuming it's already
		// a file name (without path) and just return it
		if( index < 0 )
			return fileNameAndPath;

		// Remove the first part of the string (up until the last slash)
		QString fileName = fileNameAndPath;
		fileName.remove( 0, index + 1 );
		return fileName;
	}

	///////////////////////////////////////////////////////////////////////////
	QStringList BootstrapFiberTrackingPlugin::getBootstrapFileNames( const QString directory, const QString fileName )
	{
		// Split up file name in base name and extension (if it has one)
		QStringList parts = fileName.split( "." );
		QString base = parts.at( 0 );
		QString extension = parts.count() == 2 ? parts.at( 1 ) : "";

		// Create search filter consisting of the base name, a wild card
		// and the extension. The base name has to be stripped of any
		// numeric postfixes however. We assume these are either 0 or 1.
		// We also assume no zero-padding was used
		QString filter = base;
		if( filter.endsWith( '0' ) || filter.endsWith( '1' ) )
			filter.chop( 1 );
		filter.append( "*" );
		filter.append( extension );

		// Append trailing slash to directory path
		QString dirPath = directory;
		dirPath.append( '/' );

		QDir dir( dirPath, filter );
		QStringList fileNames = dir.entryList();
		QStringList fileNamesAndPaths;
		foreach( QString name, fileNames )
		{
			QString newName;
			newName.append( dirPath );
			newName.append( name );
			fileNamesAndPaths.append( newName );
		}

		return fileNamesAndPaths;
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::saveROI()
	{
		// Get filename and path
		QString name = _form->comboSeedRegions->currentText();
		QString path = QFileDialog::getSaveFileName( 0, "Specify File Name", "." );
		if( path.isEmpty() )
			return;

		// Put them together and add VTK extension
		QString filePath = path + "/" + name + ".vtk";

		// Get seed points
		int index = _form->comboSeedRegions->currentIndex();
		vtkUnstructuredGrid * seedPoints = _seedRegions.at( index );

		// Create writer for writing points to file
		vtkUnstructuredGridWriter * writer = vtkUnstructuredGridWriter::New();
		writer->SetFileName( filePath.toAscii().data() );
		writer->SetFileTypeToBinary();
		writer->SetInput( seedPoints );
		writer->Write();

		// Clean up
		writer->Delete();
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::loadROI()
	{
		// Open file dialog to select seed region's file name
		QString path = QFileDialog::getOpenFileName( 0, "Select Seed Region", "." );
		if( path.isEmpty() )
			return;

		// Extract just file name (without path)
		QString name = path;
		int index = path.lastIndexOf( '/' );
		if( index < 0 )
			index = path.lastIndexOf( '\\' );
		name.remove( 0, index + 1 );

		// Trim extension
		index = path.lastIndexOf( '.' );
		name.truncate( index );

		// Create reader for reading seed points from file
		vtkUnstructuredGridReader * reader = vtkUnstructuredGridReader::New();
		reader->SetFileName( path.toAscii().data() );
		reader->Update();

		// Get seed points from reader
		vtkUnstructuredGrid * seedPoints = reader->GetOutput();

		// Add seed points to internal list and combo box. Note that this
		// will *not* broadcast the import to other plugins. Also, you will
		// not see the seed region in the 2D and 3D view
		_seedRegions.append( seedPoints );
		_form->comboSeedRegions->addItem( name );
		_form->comboSeedRegions->setCurrentIndex( _form->comboSeedRegions->count() - 1 );

		std::cout << "BootstrapFiberTrackingPlugin::loadROI() loaded "
				  << name.toAscii().data() << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////
	void BootstrapFiberTrackingPlugin::setupConnections()
	{
		this->connect( _form->buttonRun, SIGNAL( clicked() ), this, SLOT( run() ) );
		this->connect( _form->buttonBootstrapDirectory, SIGNAL( clicked() ), this, SLOT( openDir() ) );
		this->connect( _form->buttonLoad, SIGNAL( clicked() ), this, SLOT( loadROI() ) );
		this->connect( _form->buttonSave, SIGNAL( clicked() ), this, SLOT( saveROI() ) );
	}
}

Q_EXPORT_PLUGIN2( libBootstrapFiberTrackingPlugin, bmia::BootstrapFiberTrackingPlugin )
