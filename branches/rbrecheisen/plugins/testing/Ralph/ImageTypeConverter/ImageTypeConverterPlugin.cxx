// Includes DTI tool
#include <ImageTypeConverterPlugin.h>

// Includes VTK
#include <vtkImageShiftScale.h>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	ImageTypeConverterPlugin::ImageTypeConverterPlugin() : plugin::Plugin( "ImageTypeConverterPlugin" ),
		data::Consumer(), plugin::GUI()
	{
		// Setup UI
		_datasetBox = new QComboBox;
		_typeBox = new QComboBox;
		_typeBox->addItem( "UnsignedChar" );
		_typeBox->addItem( "UnsignedShort" );
		_typeBox->addItem( "Integer" );
		_typeBox->addItem( "Float" );
		_typeBox->addItem( "Double" );

		_button = new QPushButton( "Convert" );
        _buttonPad = new QPushButton( "Pad" );
        _buttonSave = new QPushButton( "Save to .VOL" );

		_widget = new QWidget;
		_layout = new QVBoxLayout;
		_layout->addWidget( new QLabel( "Dataset" ) );
		_layout->addWidget( _datasetBox );
		_layout->addWidget( new QLabel( "Convert to Type" ) );
		_layout->addWidget( _typeBox );
		_layout->addWidget( _button );
		_layout->addWidget( _buttonSave );
        _layout->addWidget( _buttonPad );
		_layout->addStretch();

		_widget->setLayout( _layout );

		// Setup connections
		this->connect( _button, SIGNAL( clicked() ), this, SLOT( convert() ) );
		this->connect( _buttonSave, SIGNAL( clicked() ), this, SLOT( save() ) );
        this->connect( _buttonPad, SIGNAL( clicked() ), this, SLOT( pad() ) );
	}

	///////////////////////////////////////////////////////////////////////////
	ImageTypeConverterPlugin::~ImageTypeConverterPlugin()
	{
		// Delete QT objects
		delete _widget;
		delete _layout;
		delete _button;
		delete _buttonSave;
		delete _typeBox;
		delete _datasetBox;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * ImageTypeConverterPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void ImageTypeConverterPlugin::dataSetAdded( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;

		// Add dataset name to combo box and its kind to map
		QString name = dataset->getName();
		if( dataset->getVtkImageData() != 0 )
		{
			_datasets.append( dataset->getVtkImageData() );
			_datasetBox->addItem( name );
			_kind = dataset->getKind();

			_datasetBox->setCurrentIndex( _datasetBox->count() - 1 );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void ImageTypeConverterPlugin::dataSetRemoved( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void ImageTypeConverterPlugin::dataSetChanged( data::DataSet * dataset )
	{
	}

	///////////////////////////////////////////////////////////////////////////
	void ImageTypeConverterPlugin::convert()
	{
		// Check if we have loaded datasets to transform
		if( _datasetBox->count() == 0 )
			return;

		// Get selected dataset
		vtkImageData * data = _datasets.at( _datasetBox->currentIndex() );
		vtkImageData * converted = 0;

		// Get selected target data type
		QString dataType = _typeBox->currentText();
		if( dataType == "UnsignedChar" )
		{
			double * range = data->GetScalarRange();
			vtkImageShiftScale * filter = vtkImageShiftScale::New();
			filter->ClampOverflowOn();
			filter->SetInput( data );
			filter->SetOutputScalarTypeToUnsignedChar();
			filter->SetShift( -range[0] );
			if( range[1] - range[0] > 255.0 )
				filter->SetScale( 255.0 / (range[1] - range[0]) );
			else
				filter->SetScale( 1.0 );
			filter->Update();

			converted = filter->GetOutput();
			converted->Register( 0 );
			filter->Delete();
		}
		else if( dataType == "UnsignedShort" )
		{
			double * range = data->GetScalarRange();
			vtkImageShiftScale * filter = vtkImageShiftScale::New();
			filter->ClampOverflowOn();
			filter->SetInput( data );
			filter->SetOutputScalarTypeToUnsignedShort();
			filter->SetShift( -range[0] );
			if( range[1] - range[0] > 4096.0 )
				filter->SetScale( 4096.0 / (range[1] - range[0]) );
			else
				filter->SetScale( 1.0 );
			filter->Update();

			converted = filter->GetOutput();
			converted->Register( 0 );
			filter->Delete();
		}
		else if( dataType == "Integer" )
		{
			double * range = data->GetScalarRange();
			vtkImageShiftScale * filter = vtkImageShiftScale::New();
			filter->ClampOverflowOn();
			filter->SetInput( data );
			filter->SetOutputScalarTypeToInt();
			filter->SetShift( -range[0] );
			if( range[1] - range[0] > 4096.0 )
				filter->SetScale( 4096.0 / (range[1] - range[0]) );
			else
				filter->SetScale( 1.0 );
			filter->Update();

			converted = filter->GetOutput();
			converted->Register( 0 );
			filter->Delete();
		}
		else if( dataType == "Float" )
		{
			double * range = data->GetScalarRange();
			vtkImageShiftScale * filter = vtkImageShiftScale::New();
			filter->ClampOverflowOn();
			filter->SetInput( data );
			filter->SetOutputScalarTypeToFloat();
			filter->SetShift( -range[0] );
			if( range[1] - range[0] > 4096.0 )
				filter->SetScale( 4096.0 / (range[1] - range[0]) );
			else
				filter->SetScale( 1.0 );
			filter->Update();

			converted = filter->GetOutput();
			converted->Register( 0 );
			filter->Delete();
		}
		else if( dataType == "Double" )
		{
			double * range = data->GetScalarRange();
			std::cout<<"range("<<range[0]<<","<<range[1]<<")"<<std::endl;
			vtkImageShiftScale * filter = vtkImageShiftScale::New();
			filter->ClampOverflowOn();
			filter->SetInput( data );
			filter->SetOutputScalarTypeToDouble();
			filter->SetShift( -range[0] );
			if( range[1] - range[0] > 4096.0 )
				filter->SetScale( 4096.0 / (range[1] - range[0]) );
			else
				filter->SetScale( 1.0 );
			filter->Update();

			converted = filter->GetOutput();
			converted->Register( 0 );
			filter->Delete();
		}
		else {}

		if( converted )
		{
			double bounds[6];
			converted->GetBounds( bounds );
			double * range = converted->GetScalarRange();
			QString newName = _datasetBox->currentText().append( dataType );
			data::DataSet * newDataset = new data::DataSet( newName, _kind, converted );
			this->core()->data()->addDataSet( newDataset );
			this->core()->render();
		}
	}

    ///////////////////////////////////////////////////////////////////////////
    void ImageTypeConverterPlugin::pad()
    {
        if( _datasetBox->count() == 0 )
            return;

        vtkImageData * data = _datasets.at( _datasetBox->currentIndex() );

        int dims[3];
        data->GetDimensions( dims );
        double spacing[3];
        data->GetSpacing( spacing );

        int newDims[3];
        newDims[0] = this->NextPowerOfTwo( dims[0] );
        newDims[1] = this->NextPowerOfTwo( dims[1] );
        newDims[2] = dims[2];

        int newSizeTotal = newDims[0] * newDims[1] * newDims[2];
        unsigned short * voxels = (unsigned short *) data->GetScalarPointer();

        vtkImageData * newData = vtkImageData::New();
        newData->SetOrigin( 0, 0, 0 );
        newData->SetScalarTypeToUnsignedShort();
        newData->SetNumberOfScalarComponents( 1 );
        newData->SetSpacing( spacing[0], spacing[1], spacing[2] );
        newData->SetExtent( 0, newDims[0]-1, 0, newDims[1]-1, 0, newDims[2]-1 );
        newData->AllocateScalars();

        unsigned short * newVoxels = (unsigned short *) newData->GetScalarPointer();
        for( int i = 0; i < newSizeTotal; ++i )
            newVoxels[i] = 0;

        for( int k = 0; k < dims[2]; ++k )
        {
            for( int i = 0; i < dims[1]; ++i )
            {
                for( int j = 0; j < dims[0]; ++j )
                {
                    int idx0 = k * dims[0] * dims[1] + i * dims[0] + j;
                    int idx1 = k * newDims[0] * newDims[1] + i * newDims[0] + j;
                    newVoxels[idx1] = voxels[idx0];
                }
            }
        }

        QString dataType = _typeBox->currentText();
        QString newName = _datasetBox->currentText().append( dataType );
        data::DataSet * newDataset = new data::DataSet( newName, _kind, newData );
        this->core()->data()->addDataSet( newDataset );
        this->core()->render();
    }

    ///////////////////////////////////////////////////////////////////////////
	void ImageTypeConverterPlugin::save()
	{
		QString fileName = QFileDialog::getSaveFileName( 0, "Save", "/Users/Ralph/Datasets" );
		if( fileName.isEmpty() )
			return;

		if( _datasets.empty() )
			return;

		vtkImageData * data = _datasets.at( _datasetBox->currentIndex() );
		void * voxels = data->GetScalarPointer();
		Q_ASSERT( voxels );

		int dims[3];
		data->GetDimensions( dims );
		double spacing[3];
		data->GetSpacing( spacing );
		int nrVoxels = dims[0] * dims[1] * dims[2];

		QString fileNameRaw = fileName + ".raw";
		FILE * rawFile = fopen( fileNameRaw.toStdString().c_str(), "wb" );

		if( data->GetScalarType() == VTK_UNSIGNED_CHAR )
		{
			fwrite( voxels, sizeof( unsigned char ), nrVoxels, rawFile );
		}
		else if( data->GetScalarType() == VTK_UNSIGNED_SHORT )
		{
			std::cout << "Writing unsigned short voxels" << std::endl;
			fwrite( voxels, sizeof( unsigned short ), nrVoxels, rawFile );
		}
		else if( data->GetScalarType() == VTK_FLOAT )
		{
			fwrite( voxels, sizeof( float ), nrVoxels, rawFile );
		}
		else if( data->GetScalarType() == VTK_DOUBLE )
		{
			fwrite( voxels, sizeof( double ), nrVoxels, rawFile );
		}
		else {}

		fclose( rawFile );

		QString fileNameVol = fileName + ".vol";
		FILE * volFile = fopen( fileNameVol.toStdString().c_str(), "wt" );
		fprintf( volFile, "Data.FileName = %s\n", fileNameRaw.toStdString().c_str() );
		fprintf( volFile, "Data.Type = %s\n", "raw" );
		fprintf( volFile, "Data.Dimensions = %d %d %d\n", dims[0], dims[1], dims[2] );
		fprintf( volFile, "Data.PixelSpacing = %lf %lf %lf\n", spacing[0], spacing[1], spacing[2] );
		fprintf( volFile, "Data.NrBits = %d\n", 8 * data->GetScalarSize() );
		fprintf( volFile, "Data.NrComponents = 1\n" );
		fclose( volFile );
	}

    //////////////////////////////////////////////////////////////////////
    int ImageTypeConverterPlugin::NextPowerOfTwo( int number )
    {
        int k = 2;
        if( number == 0 ) return 1;
        while( k < number )
            k *= 2;
        return k;
    }
}

Q_EXPORT_PLUGIN2( libImageTypeConverterPlugin, bmia::ImageTypeConverterPlugin )
