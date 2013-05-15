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

// Includes DTITool
#include <FibersToVoxelsPlugin.h>
#include <vtkStreamlineToVoxelDensity.h>
//#include <vtkDTIReader2.h>
#include <core/Core.h>

// Includes VTK
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkMath.h>

#include <limits.h>

// Definition of cache directory. This directory is used to
// store intermediate computation results
#define CACHE_DIRECTORY "/home/rbrecheis/datasets/tilburg"

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	FibersToVoxelsPlugin::FibersToVoxelsPlugin() :
		plugin::Plugin( "FibersToVoxelsPlugin" ),
		data::Consumer(),
		plugin::GUI()
	{
        _matrix = NULL;
        _fiberLUT = NULL;

		_widget = new QWidget;
		_fiberBox = new QComboBox;
		_datasetBox = new QComboBox;

        _densityBox = new QComboBox;
        _densityBox->addItem("2");
        _densityBox->addItem("4");
        _densityBox->addItem("8");
        _densityBox->addItem("16");
        _densityBox->addItem("32");

        _distanceMeasureBox = new QComboBox;
        _distanceMeasureBox->addItem("End Points");
        _distanceMeasureBox->addItem("Mean of Closest Points");

        _button = new QPushButton( "Voxelize" );
        _buttonLoad = new QPushButton( "Load Scores..." );
        _buttonLoadVolume = new QPushButton( "Load Volume..." );
        _buttonSave = new QPushButton( "Save Voxels..." );
        _buttonUndoTransform = new QPushButton( "Apply Transform..." );
        _buttonInsertScores = new QPushButton( "Insert and Save Scores" );
        _buttonSaveFibers = new QPushButton( "Save Fibers..." );
        _buttonInsertLUT = new QPushButton( "Update LUT" );
        _buttonComputeScoresFromDistances = new QPushButton( "Compute Distances..." );
        _buttonComputeScores = new QPushButton( "Compute Scores..." );
        _buttonInvertScores = new QPushButton( "Invert Scores..." );
        _buttonSeedPointsToText = new QPushButton( "Seed Points to Text..." );

        _sliderScores = new QSlider( Qt::Horizontal );
        _sliderScores->setRange( 0, 100 );
        _sliderScores->setValue( 0 );

        _sliderOpacity = new QSlider( Qt::Horizontal );
        _sliderOpacity->setRange( 0, 100 );
        _sliderOpacity->setValue( 100 );

		_dimEdit[0] = new QLineEdit( "0" );
		_dimEdit[1] = new QLineEdit( "0" );
		_dimEdit[2] = new QLineEdit( "0" );

		_spacingEdit[0] = new QLineEdit( "1.0" );
		_spacingEdit[1] = new QLineEdit( "1.0" );
		_spacingEdit[2] = new QLineEdit( "1.0" );

        _colorEdit[0] = new QLineEdit( "1.0" );
        _colorEdit[1] = new QLineEdit( "1.0" );
        _colorEdit[2] = new QLineEdit( "1.0" );

        _nrSeedPointsEdit = new QLineEdit("133");
        _nrIterationsEdit = new QLineEdit("250");

		_binaryBox = new QCheckBox( "Binary" );
		_binaryBox->setChecked( false );

		QHBoxLayout * dimLayout = new QHBoxLayout;
		dimLayout->addWidget( _dimEdit[0] );
		dimLayout->addWidget( _dimEdit[1] );
		dimLayout->addWidget( _dimEdit[2] );

		QHBoxLayout * spacingLayout = new QHBoxLayout;
		spacingLayout->addWidget( _spacingEdit[0] );
		spacingLayout->addWidget( _spacingEdit[1] );
		spacingLayout->addWidget( _spacingEdit[2] );

        QHBoxLayout * colorLayout = new QHBoxLayout;
        colorLayout->addWidget( _colorEdit[0] );
        colorLayout->addWidget( _colorEdit[1] );
        colorLayout->addWidget( _colorEdit[2] );

        QHBoxLayout * extraLayout = new QHBoxLayout;
        extraLayout->addWidget(_nrSeedPointsEdit);
        extraLayout->addWidget(_nrIterationsEdit);

		_layout = new QVBoxLayout;
		_layout->addWidget( new QLabel( "Volume Dimensions" ) );
		_layout->addLayout( dimLayout );
		_layout->addWidget( new QLabel( "Voxel Spacing" ) );
		_layout->addLayout( spacingLayout );
		_layout->addWidget( _binaryBox );

        QGridLayout * grid2 = new QGridLayout;
        grid2->addWidget( new QLabel( "Volumes" ), 0, 0 );
        grid2->addWidget( _datasetBox, 0, 1 );
        grid2->addWidget( new QLabel( "Fibers" ), 1, 0 );
        grid2->addWidget( _fiberBox, 1, 1 );
        grid2->addWidget(new QLabel("Distance Measure"), 2, 0);
        grid2->addWidget(_distanceMeasureBox, 2, 1);
        grid2->addWidget( new QLabel("Seed Density"), 3, 0 );
        grid2->addWidget(_densityBox, 3, 1);

        QGridLayout * grid = new QGridLayout;
        grid->addWidget( _buttonLoad, 0, 0 );
        grid->addWidget( _button, 0, 1 );
        grid->addWidget( _buttonSave, 1, 0 );
        grid->addWidget( _buttonUndoTransform, 1, 1 );
        grid->addWidget( _buttonSaveFibers, 2, 0 );
        grid->addWidget( _buttonComputeScores, 2, 1 );
        grid->addWidget( _buttonComputeScoresFromDistances, 3, 0 );
        grid->addWidget( _buttonInvertScores, 3, 1 );
        grid->addWidget( _buttonSeedPointsToText, 4, 0 );

        _layout->addLayout(grid2);
        _layout->addLayout( grid );
        _layout->addLayout(extraLayout);
        _layout->addLayout( colorLayout );
        _layout->addWidget( new QLabel( "Score Threshold" ) );
        _layout->addWidget( _sliderScores );
        _layout->addWidget( new QLabel( "Opacity" ) );
        _layout->addWidget( _sliderOpacity );
        _layout->addWidget( _buttonInsertLUT );
        _layout->addStretch();
		_widget->setLayout( _layout );

		// Connect events
		this->connect( _button, SIGNAL( clicked() ), this, SLOT( compute() ) );
        this->connect( _buttonLoad, SIGNAL( clicked() ), this, SLOT( loadScores() ) );
		this->connect( _buttonSave, SIGNAL( clicked() ), this, SLOT( save() ) );
        this->connect( _buttonSaveFibers, SIGNAL( clicked() ), this, SLOT( saveFibers() ) );
        this->connect( _buttonUndoTransform, SIGNAL( clicked() ), this, SLOT( transform() ) );
        this->connect( _buttonInsertScores, SIGNAL( clicked() ), this, SLOT( insertScores() ) );
        this->connect( _buttonInsertLUT, SIGNAL( clicked() ), this, SLOT( updateLUT() ) );
        this->connect( _buttonComputeScoresFromDistances, SIGNAL( clicked() ), this, SLOT( computeScoresFromDistances() ) );
        this->connect( _buttonComputeScores, SIGNAL( clicked() ), this, SLOT( computeScores() ) );
        this->connect( _buttonInvertScores, SIGNAL( clicked() ), this, SLOT( invertScores() ) );
        this->connect( _buttonSeedPointsToText, SIGNAL( clicked() ), this, SLOT( seedPointsToText() ) );
        this->connect( _sliderScores, SIGNAL( valueChanged(int) ), this, SLOT( scoreChanged(int) ) );
        this->connect( _sliderOpacity, SIGNAL( valueChanged(int) ), this, SLOT( opacityChanged(int) ) );
    }

	///////////////////////////////////////////////////////////////////////////
	FibersToVoxelsPlugin::~FibersToVoxelsPlugin()
	{
		// Delete QT objects
		delete _widget;
		delete _layout;
		delete _button;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * FibersToVoxelsPlugin::getGUI()
	{
		return _widget;
	}

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::undoTransform()
    {
        if( _matrix )
        {
            vtkTransform * transform = vtkTransform::New();
            transform->SetMatrix( _matrix );
            transform->Inverse();

            vtkPolyData * fibers = _fiberList.at( _fiberBox->currentIndex() );
            vtkTransformFilter * filter = vtkTransformFilter::New();
            filter->SetInput( fibers );
            filter->SetTransform( transform );
            filter->Update();

            vtkPolyData * newFibers = vtkPolyData::SafeDownCast( filter->GetOutput() );
            vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
            writer->SetFileName( "/home/rbrecheis/datasets/tilburg/kuhl280909/contrack/fibers/conTrack/fibersUndoTransform.fbs" );
            writer->SetInput( newFibers );
            writer->Write();

            filter->Delete();
            writer->Delete();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::transform()
    {
        vtkMatrix4x4 * matrix = vtkMatrix4x4::New();
        matrix->SetElement( 0, 0, 1.75 );
        matrix->SetElement( 0, 1, 0.00 );
        matrix->SetElement( 0, 2, 0.00 );
        matrix->SetElement( 0, 3, 0.00 );
        matrix->SetElement( 1, 0, 0.00 );
        matrix->SetElement( 1, 1, 1.75 );
        matrix->SetElement( 1, 2, 0.00 );
        matrix->SetElement( 1, 3, 0.00 );
        matrix->SetElement( 2, 0, 0.00 );
        matrix->SetElement( 2, 1, 0.00 );
        matrix->SetElement( 2, 2, 2.00 );
        matrix->SetElement( 2, 3, 0.00 );
        matrix->SetElement( 3, 0, 0.00 );
        matrix->SetElement( 3, 1, 0.00 );
        matrix->SetElement( 3, 2, 0.00 );
        matrix->SetElement( 3, 3, 1.00 );

        if( matrix )
        {
            vtkTransform * transform = vtkTransform::New();
            transform->SetMatrix( matrix );

            vtkPolyData * fibers = _fiberList.at( _fiberBox->currentIndex() );
            vtkTransformFilter * filter = vtkTransformFilter::New();
            filter->SetInput( fibers );
            filter->SetTransform( transform );
            filter->Update();

            QString fileName = QFileDialog::getSaveFileName( 0, "Save Fibers?", CACHE_DIRECTORY );
            if( fileName.isNull() )
                return;

            vtkPolyData * newFibers = vtkPolyData::SafeDownCast( filter->GetOutput() );
            vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
            writer->SetFileName( fileName.toStdString().c_str() );
            writer->SetInput( newFibers );
            writer->Write();

            QMessageBox::information( NULL, "Info", "Fibers saved" );

            filter->Delete();
            writer->Delete();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::compute()
	{
		// If we did not store any fibers, quit here
		if( _fiberList.size() == 0 )
			return;

		// Get selected fiber set
		int index = _fiberBox->currentIndex();
		vtkPolyData * fibers = _fiberList.at( index );
		Q_ASSERT( fibers );

		// Get dataset dimensions
		int dims[3];
		for( int i = 0; i < 3; ++i )
			dims[i] = _dimEdit[i]->text().toInt();

		// Get dataset spacing
		double spacing[3];
		for( int i = 0; i < 3; ++i )
			spacing[i] = _spacingEdit[i]->text().toDouble();

		// Get binary option
		int binary = _binaryBox->isChecked() ? 1 : 0;

		// Create streamline to voxel filter
		vtkStreamlineToVoxelDensity * filter = new vtkStreamlineToVoxelDensity;
		filter->SetBinary( binary );
		filter->SetDimensions( dims[0], dims[1], dims[2] );
		filter->SetSpacing( spacing[0], spacing[1], spacing[2] );
		filter->SetInput( fibers );

        if( _fiberScores.count() > 0 )
        {
            int count = _fiberScores.count();
            double * scores = new double[count];
            for( int i = 0; i < count; ++i )
                scores[i] = _fiberScores.at( i );
            filter->SetScores( scores, count );
            delete [] scores;
        }

		vtkImageData * voxels = filter->GetOutput();
        data::DataSet * dataset = new data::DataSet(
                _fiberBox->currentText().append( "Voxels" ), "scalar volume", voxels );
        this->core()->data()->addDataSet( dataset );
        _datasetBox->setCurrentIndex( _datasetBox->count() - 1 );

        save();
	}

	///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::load()
	{
		// Get filename for bootstrap fibers
		QString fileNameAndPath = QFileDialog::getOpenFileName( 0, "Load Bootstrap Fibers",
				CACHE_DIRECTORY, "*.fbs *.vtk" );
		if( fileNameAndPath.isNull() )
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

		// Load fibers from file
		vtkPolyDataReader * reader = vtkPolyDataReader::New();
		reader->SetFileName( fileNameAndPath.toStdString().c_str() );
		vtkPolyData * fibers = reader->GetOutput();
		reader->Update();
		fibers->Register( 0 );

		// Add fibers to list and update combobox
		_fiberList.append( fibers );
		_fiberBox->addItem( fileName );
	}

	///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::loadVolume()
	{
		//// Get filename for bootstrap fibers
		//QString fileNameAndPath = QFileDialog::getOpenFileName( 0, "Load DTI Volume",
		//		CACHE_DIRECTORY, "*.dti" );
		//if( fileNameAndPath.isNull() )
		//	return;

		//// Create base filename without extension
		//QString baseName = fileNameAndPath;
		//baseName.remove( baseName.length() - 4, 4 );

		//// Create filename without path
		//QString fileName = baseName;
		//int index = fileName.lastIndexOf( '/' );
		//if( index < 0 )
		//	index = fileName.lastIndexOf( '\\' );
		//fileName.remove( 0, index + 1 );

		//// Load volume from file
		//vtkDTIReader2 * reader = vtkDTIReader2::New();
		//reader->SetFileName( fileNameAndPath.toStdString().c_str() );
		//vtkImageData * volume = reader->GetOutput();
		//reader->Update();
		//volume->Register( 0 );

		//// Add fibers to list and update combobox
		//_datasetList.append( volume );
		//_datasetBox->addItem( fileName );
	}

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::invertScores()
    {
        QString fileNameAndPath = QFileDialog::getOpenFileName( 0, "Load fiber scores",
                CACHE_DIRECTORY, "*.txt" );
        if( fileNameAndPath.isNull() )
            return;
        QFile file( fileNameAndPath );
        file.open( QIODevice::ReadOnly | QIODevice::Text );
        if( file.error() )
            std::cout << "FibersToVoxelsPlugin::invertScores() cannot open file" << std::endl;
        QTextStream stream( & file );
        QList< double > scores;
        QString line = stream.readLine();
        while( ! line.isNull() )
        {
            QStringList tokens = line.split( " " );
            scores.append( tokens.at( 1 ).toDouble() );
            line = stream.readLine();
        }
        file.close();
        QString outFileName = fileNameAndPath.append("_new.txt");
        QFile outFile(outFileName);
        outFile.open(QIODevice::WriteOnly | QIODevice::Text);
        if(outFile.error())
            std::cout << "FibersToVoxelsPlugin::invertScores() cannot open file for writing" << std::endl;
        QTextStream outStream(&outFile);
        for(int i = 0; i < scores.count(); ++i)
            outStream << i << " " << scores.at(i) << "\n";
        outFile.close();
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::loadScores()
    {
        // Get filename for fiber scores
        QString fileNameAndPath = QFileDialog::getOpenFileName( 0, "Load fiber scores",
                CACHE_DIRECTORY, "*.txt" );
        if( fileNameAndPath.isNull() )
            return;

        QFile file( fileNameAndPath );
        file.open( QIODevice::ReadOnly | QIODevice::Text );
        if( file.error() )
            std::cout << "FibersToVoxelsPlugin::loadScores() cannot open file" << std::endl;
        QFileInfo info( fileNameAndPath );
        QTextStream stream( & file );

        QList< double > scores;
        QString line = stream.readLine();
        while( ! line.isNull() )
        {
            QStringList tokens = line.split( " " );
            scores.append( tokens.at( 1 ).toDouble() );
            line = stream.readLine();
        }

        file.close();

        double range[2];
        range[0] =  99999999.0;
        range[1] = -99999999.0;
        for( int i = 0; i < scores.count(); ++i )
        {
            if( scores.at( i ) < range[0] ) range[0] = scores.at( i );
            if( scores.at( i ) > range[1] ) range[1] = scores.at( i );
        }

        _fiberScores.clear();

        for( int i = 0; i < scores.count(); ++i )
        {
            double score = (scores.at( i ) - range[0]) / (range[1] - range[0]);
            _fiberScores.append( score );

            std::cout << "score " << score << std::endl;
        }

        insertScores();
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::insertScores()
    {
        if( _fiberScores.count() == 0 || _fiberList.count() == 0 )
        {
            std::cout << "insertScores() no scores" << std::endl;
            return;
        }

        vtkPolyData * fibers = _fiberList.at( _fiberBox->currentIndex() );

        vtkDoubleArray * scores = vtkDoubleArray::New();
        scores->SetNumberOfComponents( 1 );
        scores->Allocate( fibers->GetNumberOfPoints() );
        scores->SetName( "scores" );

        vtkPointData * fiberPointData = fibers->GetPointData();
        fiberPointData->SetScalars( scores );
        scores->Delete();

        vtkIdType nrPtIds, * ptIds = NULL;
        vtkPoints * points = fibers->GetPoints();
        vtkCellArray * cells = fibers->GetLines();
        vtkDataArray * dataArray = fiberPointData->GetScalars();

        int cellIdx = 0;
        cells->InitTraversal();
        while( cells->GetNextCell( nrPtIds, ptIds ) )
        {
            for( int i = 0; i < nrPtIds; ++i )
                dataArray->InsertTuple1( ptIds[i], _fiberScores.at( cellIdx ) );
            cellIdx++;
        }

        fibers->GetPointData()->SetActiveScalars( "scores" );
        fibers->Update();
        fibers->UpdateData();

        updateLUT();

        //saveFibers();
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::computeScores()
    {
        QString fileName = QFileDialog::getSaveFileName(0, "Save Scores", CACHE_DIRECTORY);
        if(fileName.isEmpty())
            return;

        vtkPolyData * fibers = _fiberList.at(_fiberBox->currentIndex());
        vtkImageData * voxelCounts = _datasetList.at(_datasetBox->currentIndex());
        unsigned short * voxels = (unsigned short *) voxelCounts->GetScalarPointer();

        int dims[3];
        voxelCounts->GetDimensions(dims);
        double spacing[3];
        voxelCounts->GetSpacing(spacing);

        vtkIdType nrPtIds, * ptIds = 0;
        vtkPoints * points = fibers->GetPoints();
        vtkCellArray * lines = fibers->GetLines();

        QList<int> ids;
        QList<unsigned long> scores;
        unsigned long scoreMin = ULONG_MAX;
        unsigned long scoreMax = 0;
        int cellId = 0;

        lines->InitTraversal();
        while(lines->GetNextCell(nrPtIds, ptIds))
        {
            unsigned long score = 0;
            int prevIdx = -1;

            for(int i = 0; i < nrPtIds; ++i)
            {
                double P[3];
                points->GetPoint(ptIds[i], P);

                //std::cout << ptIds[i] << " ";

                int x = (int) floor(P[0] / spacing[0]);
                int y = (int) floor(P[1] / spacing[1]);
                int z = (int) floor(P[2] / spacing[2]);

                int index = z * dims[0] * dims[1] + y * dims[0] + x;
                if(index == prevIdx)
                    continue;

                unsigned short value = voxels[index];
                score += value;

                prevIdx = index;
            }

            //std::cout << std::endl;

            if(score < scoreMin) scoreMin = score;
            if(score > scoreMax) scoreMax = score;
            scores.append(score);
            ids.append(cellId);

            std::cout << "processing fiber " << cellId << " with score " << score << std::endl;
            cellId++;
        }

        std::cout << "scoreMin " << scoreMin << " scoreMax " << scoreMax << std::endl;

        // store scores globally

        _fiberScores.clear();
        double range = scoreMax - scoreMin;
        for(int i = 0; i < scores.count(); ++i)
        {
            double value = (double) (scores.at(i) - scoreMin) / range;
            _fiberScores.append(value);
        }

        // save scores to TXT

        FILE * f = fopen(fileName.toStdString().c_str(), "wt");
        for(int i = 0; i < _fiberScores.count(); ++i)
            fprintf(f, "%d %lf\n", ids.at(i), _fiberScores.at(i));
        fclose(f);

        // insert scores into fiber polydata

        insertScores();
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::seedPointsToText()
    {
        unsigned char * voxels = (unsigned char *)
                _datasetList.at(_datasetBox->currentIndex())->GetScalarPointer();
        Q_ASSERT(voxels);
        int dims[3] = {
            _dimEdit[0]->text().toInt(),
            _dimEdit[1]->text().toInt(), _dimEdit[2]->text().toInt()};
        double spacing[3] = {
            _spacingEdit[0]->text().toDouble(),
            _spacingEdit[1]->text().toDouble(), _spacingEdit[2]->text().toDouble()};
        int density = _densityBox->currentText().toInt();
        QList<Point> points;
        for(int k = 0; k < dims[2]; ++k)
        {
            for(int i = 0; i < dims[1]; ++i)
            {
                for(int j = 0; j < dims[0]; ++j)
                {
                    int idx = k * dims[0] * dims[1] + i * dims[0] + j;
                    if(voxels[idx])
                    {
                        Point P;
                        P.x = j * spacing[0];
                        P.y = i * spacing[1];
                        P.z = k * spacing[2];
                        points.append(P);
                    }
                }
            }
        }
        Q_ASSERT(points.count() > 0);
        int N = points.count();
        double offsetX = spacing[0] / density;
        double offsetY = spacing[1] / density;
        double offsetZ = spacing[2] / density;
        for(int i = 0; i < N; ++i)
        {
            const Point & P1 = points.at(i);
            for(int j = 1; j < density; ++j)
            {
                Point P2;
                P2.x = P1.x + j * offsetX;
                P2.y = P1.y + j * offsetY;
                P2.z = P1.z + j * offsetZ;
                points.append(P2);
            }
        }
        QString fileName = QFileDialog::getSaveFileName(0, "Save Seed Points", CACHE_DIRECTORY);
        if(fileName.isEmpty())
            return;
        FILE * f = fopen(fileName.toStdString().c_str(), "wt");
        Q_ASSERT(f);
        for(int i = 0; i < points.count(); ++i)
        {
            const Point & P = points.at(i);
            fprintf(f, "%f %f %f\n", P.x, P.y, P.z);
        }
        fclose(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::computeScoresFromDistances()
    {
        QString fileName = QFileDialog::getOpenFileName( 0, "Load Bfloat fibers", CACHE_DIRECTORY, "*.Bfloat" );
        if( fileName.isNull() )
            return;
        QString fileNameOut = fileName;
        fileNameOut.resize(fileName.length() - 7);
        if(_distanceMeasureBox->currentIndex() == 0)
            fileNameOut.append("_endpoint_scores.txt");
        else
            fileNameOut.append("_meanclosestpoint_scores.txt");
        QList<QList<QList<Point> > > totalFibers;
        int nrSeedPoints = _nrSeedPointsEdit->text().toInt();
        int nrPoints = 0;
        int nrIters  = _nrIterationsEdit->text().toInt();
        int seedIdx  = 0;
        int tmpIdx   = 0;
        std::cout << "Processing Bfloat..." << std::endl;
        std::ifstream f(fileName.toStdString().c_str(), ios::in | ios::binary );
        std::ofstream ff(fileNameOut.toStdString().c_str(), ios::out );
        nrPoints = (int) getFloat32(f);
        seedIdx  = (int) getFloat32(f);
        while(!f.eof())
        {
            // read all fibers for current seed point
            QList<QList<Point> > fibers;
            for(int i = 0; i < nrIters; ++i)
            {
                QList<Point> fiber;
                for(int j = 0; j < nrPoints; ++j)
                {
                    Point p;
                    p.x = getFloat32(f);
                    p.y = getFloat32(f);
                    p.z = getFloat32(f);
                    fiber.append(p);
                }
                fibers.append(fiber);
                if(f.eof())
                    break;
                nrPoints = (int) getFloat32(f);
                seedIdx  = (int) getFloat32(f);
            }
            // compute pair-wise distances between fibers
            std::cout << "computing scores " << tmpIdx << " out of " << nrSeedPoints << std::endl;
            QList<double> scores = computeNormalizedSumOfPairwiseDistances(fibers);
            // print scores to file
            writeScoresToFile(tmpIdx * nrIters, scores, ff);
            // add scores to list
            totalFibers.append(fibers);
            tmpIdx++;
        }
        f.close();
        ff.close();
    }

    ///////////////////////////////////////////////////////////////////////////
    QList<double> FibersToVoxelsPlugin::computeNormalizedSumOfPairwiseDistances(QList<QList<Point> > & fibers)
    {
        int N = fibers.count();
        int measure = _distanceMeasureBox->currentIndex();
        // initialize distance matrix
        double ** matrix = new double*[N];
        for(int i = 0; i < N; ++i)
        {
            matrix[i] = new double[N];
            for(int j = 0; j < N; ++j)
                matrix[i][j] = 0.0;
        }
        // compute distance matrix
        for(int i = 0; i < N; ++i)
        {
            QList<Point> fiberA = fibers.at(i);
            for(int j = 0; j < i; ++j)
            {
                QList<Point> fiberB = fibers.at(j);
                double d = 0.0;
                switch(measure)
                {
                case 0:
                    d = computeEndPointDistance(fiberA, fiberB);
                    break;
                case 1:
                    d = computeMeanOfClosestPointDistance(fiberA, fiberB);
                    break;
                default:
                    break;
                }
                if(d < 0.0)
                    d = 0.0;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        // find central fiber index
        double minSum = VTK_DOUBLE_MAX;
        int idx = -1;
        for(int i = 0; i < N; ++i)
        {
            double sum = 0.0;
            for(int j = 0; j < N; ++j)
                sum += matrix[i][j];
            if(sum < minSum)
            {
                minSum = sum;
                idx = i;
            }
        }
        // find maximum distance to central fiber
        double distMax = 0.0;
        for(int i = 0; i < N; ++i)
        {
            double d = matrix[idx][i];
            if(d > distMax)
                distMax = d;
        }
        // make list of fiber scores
        QList<double> scores;
        for(int i = 0; i < N; ++i)
        {
            double d = matrix[idx][i] / distMax;
            scores.append(1.0 - d);
        }
        return scores;
    }

    ///////////////////////////////////////////////////////////////////////////
    double FibersToVoxelsPlugin::computeMeanOfClosestPointDistance(QList<Point> & fiberA, QList<Point> & fiberB)
    {
        double d = 0.0;
        for(int i = 0; i < fiberA.count(); ++i)
            d += computeClosestPointDistance(fiberA.at(i), fiberB);
        for(int i = 0; i < fiberB.count(); ++i)
            d += computeClosestPointDistance(fiberB.at(i), fiberA);
        d = d / (fiberA.count() + fiberB.count());
        return d;
    }

    ///////////////////////////////////////////////////////////////////////////
    double FibersToVoxelsPlugin::computeEndPointDistance(QList<Point> & fiberA, QList<Point> & fiberB)
    {
        const Point & pA = fiberA.at(0);
        const Point & qA = fiberA.at(fiberA.count() - 1);
        const Point & pB = fiberB.at(0);
        const Point & qB = fiberB.at(fiberB.count() - 1);
        double d1 = computeDistance(pA, pB) + computeDistance(pA, qB);
        double d2 = computeDistance(qA, qB) + computeDistance(qA, pB);
        double d  = std::min(d1, d2);
        return d;
    }

    ///////////////////////////////////////////////////////////////////////////
    double FibersToVoxelsPlugin::computeClosestPointDistance(const Point & p, QList<Point> & fiber)
    {
        double dmin = VTK_DOUBLE_MAX;
        for(int i = 0; i < fiber.count(); ++i)
        {
            Point  pp = fiber.at(i);
            double d = computeDistance(p, pp);
            if(d < dmin)
                dmin = d;
        }
        return dmin;
    }

    ///////////////////////////////////////////////////////////////////////////
    double FibersToVoxelsPlugin::computeDistance(const Point & pA, const Point & pB)
    {
        double xx = (pA.x - pB.x) * (pA.x - pB.x);
        double yy = (pA.y - pB.y) * (pA.y - pB.y);
        double zz = (pA.z - pB.z) * (pA.z - pB.z);
        double d = sqrt(xx + yy + zz);
        return d;
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::writeScoresToFile(int index, QList<double> & scores, std::ofstream & f)
    {
        int writeIndex = index;
        for(int i = 0; i < scores.count(); ++i, ++writeIndex)
            f << writeIndex << " " << scores.at(i) << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::scoreChanged( int value )
    {
        if( _fiberLUT )
        {
            double color[3];
            for( int i = 0; i < 3; ++i )
                color[i] = _colorEdit[i]->text().toDouble();
            double opacity = _sliderOpacity->value() / 100.0;
            double score = value / 100.0;

            int scoreIdx = (int) floor( score * 255 );
            for( int i = 0; i < 256; ++i )
            {
                _fiberLUT->SetTableValue( i, 0.0, 0.0, 0.0, 0.0 );
                if( i > scoreIdx )
                    _fiberLUT->SetTableValue( i, color[0], color[1], color[2], opacity );
            }

            _fiberLUT->Build();

            this->core()->render();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::opacityChanged( int value )
    {
        if( _fiberLUT )
        {
            double color[3];
            for( int i = 0; i < 3; ++i )
                color[i] = _colorEdit[i]->text().toDouble();
            double score = _sliderScores->value() / 100.0;
            double opacity = value / 100.0;

            int scoreIdx = (int) floor( score * 255 );
            for( int i = 0; i < 256; ++i )
            {
                _fiberLUT->SetTableValue( i, 0.0, 0.0, 0.0, 0.0 );
                if( i > scoreIdx )
                    _fiberLUT->SetTableValue( i, color[0], color[1], color[2], opacity );
            }

            _fiberLUT->Build();

            this->core()->render();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::updateLUT()
    {
        if( _fiberLUT == NULL )
        {
            _fiberLUT = vtkLookupTable::New();
            _fiberLUT->SetTableRange( 0, 1 );
            _fiberLUT->SetNumberOfColors( 256 );

            for( int i = 0; i < 256; ++i )
                _fiberLUT->SetTableValue( i, 1.0, 1.0, 1.0, 1.0 );

            _fiberLUT->Build();

            data::DataSet * dataset = new data::DataSet( "ScoreLUT", "transfer function", _fiberLUT );
            this->core()->data()->addDataSet( dataset );
        }

        double color[3];
        for( int i = 0; i < 3; ++i )
            color[i] = _colorEdit[i]->text().toDouble();
        double score = _sliderScores->value() / 100.0;
        double opacity = _sliderOpacity->value() / 100.0;

        int scoreIdx = (int) floor( score * 255 );
        for( int i = 0; i < 256; ++i )
        {
            _fiberLUT->SetTableValue( i, 0.0, 0.0, 0.0, 0.0 );
            if( i > scoreIdx )
                _fiberLUT->SetTableValue( i, color[0], color[1], color[2], opacity );
        }

        _fiberLUT->Build();
        this->core()->render();
    }

    ///////////////////////////////////////////////////////////////////////////
    void FibersToVoxelsPlugin::saveFibers()
    {
        QString fileName = QFileDialog::getSaveFileName( 0, "Save Fibers?", CACHE_DIRECTORY );
        if( fileName.isNull() )
            return;

        vtkPolyData * fibers = _fiberList.at( _fiberBox->currentIndex() );
        vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
        writer->SetFileName( fileName.toStdString().c_str() );
        writer->SetInput( fibers );
        writer->Write();
        writer->Delete();

        if(_fiberScores.count() > 0)
        {
            QString fileNameScores = fileName;
            fileNameScores.remove(fileName.length() - 5, 4);
            fileNameScores.append(".txt");
            QFile file(fileNameScores);
            file.open(QIODevice::WriteOnly | QIODevice::Text);
            QTextStream stream(&file);
            for(int i = 0; i < _fiberScores.count(); ++i)
                stream << i << " " << _fiberScores.at(i) << "\n";
            file.close();
            QString fileNameHeader = fileName;
            fileNameHeader.remove(fileName.length() - 5, 4);
            fileNameHeader.append("_header.txt");
            QFile fileHeader(fileNameHeader);
            fileHeader.open(QIODevice::WriteOnly | QIODevice::Text);
            QTextStream streamHeader(&fileHeader);
            streamHeader << fileName << "\n";
            streamHeader << fileNameScores << "\n";
            fileHeader.close();
        }

        QMessageBox::information( NULL, "Info", "Fibers saved" );
    }

    ///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::save()
	{
		// Get filename
		QString fileName = QFileDialog::getSaveFileName( 0, "Save Voxelized Fibers", CACHE_DIRECTORY );
		if( fileName.isNull() )
			return;

		// Get selected volume dataset to obtain dimensions and spacing
		int index = _datasetBox->currentIndex();
		vtkImageData * volume = _datasetList.at( index );
		Q_ASSERT( volume );

		// Get dimensions and voxel spacing
		int dims[3], nrVoxels;
		volume->GetDimensions( dims );
		nrVoxels = dims[0] * dims[1] * dims[2];
		double spacing[3];
		volume->GetSpacing( spacing );

		// Get voxels
		void * voxels = volume->GetScalarPointer();
		int nrBytes = volume->GetScalarSize();

		// Save voxel values
		QString fileNameRaw = fileName + ".raw";
		FILE * fileRaw = fopen( fileNameRaw.toStdString().c_str(), "wb" );
		fwrite( voxels, nrBytes, nrVoxels, fileRaw );
		fclose( fileRaw );

		// Save voxel .vol file
		QString fileNameVol = fileName + ".vol";
		FILE * fileVol = fopen( fileNameVol.toStdString().c_str(), "wt" );
		fprintf( fileVol, "Data.FileName = %s\n", fileNameRaw.toStdString().c_str() );
		fprintf( fileVol, "Data.Type = raw\n" );
		fprintf( fileVol, "Data.Dimensions = %d %d %d\n", dims[0], dims[1], dims[2] );
		fprintf( fileVol, "Data.PixelSpacing = %lf %lf %lf\n", spacing[0], spacing[1], spacing[2] );
		fprintf( fileVol, "Data.NrBits = %d\n", 8 * nrBytes );
		fprintf( fileVol, "Data.NrBits = 1\n" );
		fclose( fileVol );

        QMessageBox::information( NULL, "Info", "Voxelized fibers saved" );
	}

    ///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::dataSetAdded( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;

		// Check if we're dealing with bootstrap fibers
		if( dataset->getKind() == "bootstrapFibers" || dataset->getKind() == "fibers" || dataset->getKind() == "polydata" )
		{
			// Check that we have polydata
			vtkPolyData * fibers = dataset->getVtkPolyData();
			if( ! fibers )
			{
				std::cout << "FibersToVoxelsPlugin::dataSetAdded() ";
				std::cout << "dataset is not vtkPolyData!" << std::endl;
				return;
			}

            _fiberList.append( fibers );
			_fiberBox->addItem( dataset->getName() );
		}
		else if( dataset->getKind() == "DTI" || dataset->getKind() == "scalar volume" )
		{
			// Check that we have image data
			vtkImageData * volume = dataset->getVtkImageData();
			if( ! volume )
			{
				std::cout << "FibersToVoxelsPlugin::dataSetAdded() ";
				std::cout << "dataset is not vtkImageData!" << std::endl;
				return;
			}

			// Add volume to list and its name to the combobox
			_datasetList.append( volume );
			_datasetBox->addItem( dataset->getName() );

			// Add volume dimensions to edit fields
			int dims[3];
			volume->GetDimensions( dims );
			_dimEdit[0]->setText( QString( "%1" ).arg( dims[0] ) );
			_dimEdit[1]->setText( QString( "%1" ).arg( dims[1] ) );
			_dimEdit[2]->setText( QString( "%1" ).arg( dims[2] ) );

			// Add voxel spacing to edit fields
			double spacing[3];
			volume->GetSpacing( spacing );
			_spacingEdit[0]->setText( QString( "%1" ).arg( spacing[0] ) );
			_spacingEdit[1]->setText( QString( "%1" ).arg( spacing[1] ) );
			_spacingEdit[2]->setText( QString( "%1" ).arg( spacing[2] ) );

            // Get transformation matrix (if it exists)
            vtkObject * obj = NULL;
            dataset->getAttributes()->getAttribute("transformation matrix", obj);
            if( obj )
            {
                _matrix = vtkMatrix4x4::SafeDownCast( obj );
            }
		}
		else {}
	}

	///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::dataSetRemoved( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;
	}

	///////////////////////////////////////////////////////////////////////////
	void FibersToVoxelsPlugin::dataSetChanged( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;
	}
}

Q_EXPORT_PLUGIN2( libFibersToVoxelsPlugin, bmia::FibersToVoxelsPlugin )
