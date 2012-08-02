#include <UncertaintyVisPlugin.h>
#include <core/Core.h>
#include <data/DataSet.h>
#include <data/Manager.h>

//#include <vtkConfidenceIntervalMapper.h>
#include <vtkFiberConfidenceMapper.h>
#include <vtkConfidenceHistogram.h>
#include <vtkConfidenceInterval.h>
#include <vtkConfidenceTable.h>
#include <vtkConfidenceIntervalProperties.h>
#include <vtkWidgetRepresentation.h>
#include <vtkMedicalCanvas.h>
#include <vtkROIWidget.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkActor.h>

#include <vtkDTIComponentReader.h>
#include <vtkTransformFilter.h>
#include <vtkImageReader2.h>
#include <vtkTransform.h>
#include <vtkImageWriter.h>

#include <QConfidenceHistogramWidget.h>

#include <QPushButton>
#include <QBoxLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QColorDialog>
#include <QSlider>
#include <QCheckBox>
#include <QComboBox>
#include <QFrame>
#include <QFile>
#include <QDir>
#include <QLabel>
#include <QLineEdit>

#include <string>
#include <sstream>

#include <assert.h>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	UncertaintyVisPlugin::UncertaintyVisPlugin() : plugin::AdvancedPlugin( "UncertaintyVisPlugin" )
	{
		this->selectedInterval = -1;
		this->initialized = false;
		this->actor = 0;
	}

	///////////////////////////////////////////////////////////////////////////
	UncertaintyVisPlugin::~UncertaintyVisPlugin()
	{
	}

	///////////////////////////////////////////////////////////////////////////
	vtkProp * UncertaintyVisPlugin::getVtkProp()
	{
		// Apparently getVtkProp() can be called multiple times by
		// the plugin manager, so prevent duplicate instantiations
		// (same holds for GUI widget)

		if( this->actor )
			return this->actor;

		this->histogram = vtkConfidenceHistogram::New();
		this->histogram->SetNumberOfBins( 32 );

		this->interval = vtkConfidenceInterval::New();
		this->interval->SetRange( 0.0f, 1.0f );
		this->interval->SetNumberOfIntervals( 2 );
		this->interval->SetSubdivisionToEqualWidth();
		this->interval->SetHistogram( this->histogram );
		this->interval->GetProperties()->SetColor( 0, 0.0f, 1.0f, 0.0f );
		this->interval->GetProperties()->SetColor( 1, 0.0f, 0.0f, 1.0f );
		this->interval->GetProperties()->SetOutlineColor( 0, 0.7f, 0.7f, 0.7f );
		this->interval->GetProperties()->SetOutlineColor( 1, 0.7f, 0.7f, 0.7f );
		this->interval->GetProperties()->SetBlurringEnabled( false );
		this->interval->GetProperties()->SetActivePropertyToOpacity();

		std::string str = this->interval->WriteToString();
		this->interval->ReadFromString( str );

		this->mapper = vtkFiberConfidenceMapper::New();
		this->mapper->SetInterval( this->interval );

		this->roiWidget = vtkROIWidget::New();
		this->roiWidget->SetMapper( this->mapper );

		this->actor = vtkActor::New();
		return this->actor;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * UncertaintyVisPlugin::getGUI()
	{
		if( this->initialized )
			return this->widget;

		// Check box for enabling/disabling ROI widget

		this->roiCheckBox = new QCheckBox( "Enable ROI" );
		this->roiCheckBox->setChecked( false );
		this->connect( this->roiCheckBox, 
				SIGNAL( toggled( bool ) ), this, SLOT( roiEnabled( bool ) ) );

		// Check box for enabling/disabling inversion of loaded confidence table

		this->invertCheckBox = new QCheckBox( "Invert table" );
		this->invertCheckBox->setChecked( true );

		// Push button for loading header

		this->loadHeaderButton = new QPushButton( "Load header..." );
		this->connect( this->loadHeaderButton, 
				SIGNAL( clicked() ), this, SLOT( loadHeader() ) );

		// Combo box for selecting active interval property

		QLabel * activePropertyLabel = new QLabel( "Active property" );
		this->activePropertyComboBox = new QComboBox;
		this->activePropertyComboBox->addItem( "Opacity" );
		this->activePropertyComboBox->addItem( "Outline opacity" );
		this->activePropertyComboBox->addItem( "Outline thickness" );
		this->activePropertyComboBox->addItem( "Dilation" );
		this->activePropertyComboBox->addItem( "Checker size" );
		this->activePropertyComboBox->addItem( "Hole size" );
		this->activePropertyComboBox->addItem( "Blurring radius" );
		this->activePropertyComboBox->addItem( "Blurring brightness" );
		this->activePropertyComboBox->addItem( "Noise frequency" );
		this->activePropertyComboBox->setCurrentIndex( 0 );
		this->connect( this->activePropertyComboBox,
				SIGNAL( currentIndexChanged( const QString ) ), this, SLOT( activePropertyChanged( const QString ) ) );

		// Combo box for selecting render mode (streamline, solid, checker board)

		QLabel * renderModeLabel = new QLabel( "Render mode" );
		this->renderModeComboBox = new QComboBox;
		this->renderModeComboBox->addItem( "Solid" );
		this->renderModeComboBox->addItem( "Checker board" );
		this->renderModeComboBox->addItem( "Holes" );
		this->renderModeComboBox->setCurrentIndex( 0 );
		this->connect( this->renderModeComboBox,
				SIGNAL( currentIndexChanged( const QString ) ), this, SLOT( renderModeChanged( const QString ) ) );

		// Check box for enabling/disabling blurring

		this->blurringCheckBox = new QCheckBox( "Enable blurring" );
		this->blurringCheckBox->setChecked( false );
		this->connect( this->blurringCheckBox, 
				SIGNAL( toggled( bool ) ), this, SLOT( blurringEnabled( bool ) ) );

		// Check box for enabling/disabling noise

		this->noiseCheckBox = new QCheckBox( "Enable noise" );
		this->noiseCheckBox->setChecked( false );
		this->connect( this->noiseCheckBox,
				SIGNAL( toggled( bool ) ), this, SLOT( noiseEnabled( bool ) ) );

		// Histogram widget

		this->histogramWidget = new QConfidenceHistogramWidget;
		this->connect( this->histogramWidget,
				SIGNAL( intervalChanged() ), this, SLOT( intervalChanged() ) );
		this->connect( this->histogramWidget,
				SIGNAL( intervalSelected( int ) ), this, SLOT( intervalSelected( int ) ) );

		// Slider for histogram opacity

		QLabel * histogramOpacityLabel = new QLabel( "Histogram opacity" );
		this->histogramOpacitySlider = new QSlider( Qt::Horizontal );
		this->histogramOpacitySlider->setRange( 0, 100 );
		this->histogramOpacitySlider->setValue( 50 );
		this->connect( this->histogramOpacitySlider,
				SIGNAL( valueChanged( int ) ), this, SLOT( histogramOpacityChanged( int ) ) );

		// Button for selecting interval color and color patch to display it

		this->selectColorButton = new QPushButton( "Color..." );
		this->connect( this->selectColorButton,
				SIGNAL( clicked() ), this, SLOT( colorSelected() ) );
		this->colorPatch = new QFrame;
		this->colorPatch->setMinimumWidth( 50 );
		this->colorPatch->setAutoFillBackground( true );
		this->colorPatch->setPalette( QPalette( Qt::white ) );

		// Button for selecting interval outline color and color patch to display it

		this->selectOutlineColorButton = new QPushButton( "Outline color..." );
		this->connect( this->selectOutlineColorButton,
				SIGNAL( clicked() ), this, SLOT( outlineColorSelected() ) );
		this->outlineColorPatch = new QFrame;
		this->outlineColorPatch->setMinimumWidth( 50 );
		this->outlineColorPatch->setAutoFillBackground( true );
		this->outlineColorPatch->setPalette( QPalette( Qt::white ) );

		// Combobox for selecting auto-coloring scheme

		this->autoColorComboBox = new QComboBox;
		this->autoColorComboBox->addItem( "Custom" );
		this->autoColorComboBox->addItem( "Light to dark" );
		this->autoColorComboBox->addItem( "Dark to light" );
		this->autoColorComboBox->addItem( "Decreasing saturation" );
		this->autoColorComboBox->addItem( "Increasing saturation" );
		this->autoColorComboBox->setCurrentIndex( 0 );
		this->connect( this->autoColorComboBox,
			SIGNAL( currentIndexChanged( const QString ) ), this, SLOT( autoColorSchemeSelected( const QString ) ) );

		this->selectBaseColorButton = new QPushButton( "Base color..." );
		this->connect( this->selectBaseColorButton,
			SIGNAL( clicked() ), this, SLOT( baseColorSelected() ) );

		this->baseColorPatch = new QFrame;
		this->baseColorPatch->setMinimumWidth( 50 );
		this->baseColorPatch->setAutoFillBackground( true );
		this->baseColorPatch->setPalette( QPalette( Qt::white ) );

		this->applyStaircaseButton = new QPushButton( "Staircase" );
		this->connect( this->applyStaircaseButton,
			SIGNAL( clicked() ), this, SLOT( staircaseApplied() ) );

		this->applyInvertedStaircaseButton = new QPushButton( "Inverted staircase" );
		this->connect( this->applyInvertedStaircaseButton,
			SIGNAL( clicked() ), this, SLOT( invertedStaircaseApplied() ) );

		this->applyEqualizeToSelectedButton = new QPushButton( "Equalize to selected" );
		this->connect( this->applyEqualizeToSelectedButton,
			SIGNAL( clicked() ), this, SLOT( equalizeToSelectedApplied() ) );

		this->disableSelectedButton = new QPushButton( "Enable/disable" );
		this->connect( this->disableSelectedButton,
			SIGNAL( clicked() ), this, SLOT( disableSelected() ) );

		QLabel * subdivisionLabel = new QLabel( "Subdivision" );
		this->subdivisionComboBox = new QComboBox;
		this->subdivisionComboBox->addItem( "Equal width" );
		this->subdivisionComboBox->addItem( "Equal histogram area" );
		this->subdivisionComboBox->addItem( "Custom" );
		this->subdivisionComboBox->setCurrentIndex( 0 );
		this->connect( this->subdivisionComboBox,
			SIGNAL( currentIndexChanged( const QString ) ), this, SLOT( subdivisionChanged( const QString ) ) );

		QLabel * numberOfIntervalsLabel = new QLabel( "Nr. intervals" );
		this->numberOfIntervalsComboBox = new QComboBox;
		this->numberOfIntervalsComboBox->addItem( "2" );
		this->numberOfIntervalsComboBox->addItem( "3" );
		this->numberOfIntervalsComboBox->addItem( "4" );
		this->numberOfIntervalsComboBox->addItem( "5" );
		this->numberOfIntervalsComboBox->addItem( "6" );
		this->numberOfIntervalsComboBox->setCurrentIndex( 1 );
		this->connect( this->numberOfIntervalsComboBox,
			SIGNAL( currentIndexChanged( int ) ), this, SLOT( numberOfIntervalsChanged( int ) ) );

		this->savePropertiesButton = new QPushButton( "Save properties..." );
		this->connect( this->savePropertiesButton,
			SIGNAL( clicked() ), this, SLOT( saveProperties() ) );

		this->loadPropertiesButton = new QPushButton( "Load properties..." );
		this->connect( this->loadPropertiesButton,
			SIGNAL( clicked() ), this, SLOT( loadProperties() ) );

		this->rangeLabel = new QLabel( "0.0" );
		this->rangeSlider = new QSlider( Qt::Horizontal );
		this->rangeSlider->setRange( 0, 100 );
		this->rangeSlider->setValue( 0 );
		this->connect( this->rangeSlider, SIGNAL( sliderReleased() ), this, SLOT( intervalSliderReleased() ) );
		this->connect( this->rangeSlider, SIGNAL( valueChanged( int ) ), this, SLOT( intervalRangeChanged( int ) ) );

		QPushButton * transformButton = new QPushButton( "Load transform..." );
		this->connect( transformButton, SIGNAL( clicked() ), this, SLOT( loadTransform() ) );

		QPushButton * screenshotButton = new QPushButton( "Make screenshot" );
		this->connect( screenshotButton, SIGNAL( clicked() ), this, SLOT( makeScreenshot() ) );

		QPushButton * loadDATButton = new QPushButton( "Convert DAT" );
		this->connect( loadDATButton, SIGNAL( clicked() ), this, SLOT( loadDAT() ) );

		QPushButton * startTimingButton = new QPushButton( "Start timing" );
		this->connect( startTimingButton, SIGNAL( clicked() ), this, SLOT( startTiming() ) );

		// Setup layouts

		QGridLayout * layout1 = new QGridLayout;
		layout1->addWidget( renderModeLabel, 0, 0 );
		layout1->addWidget( this->renderModeComboBox, 0, 1 );
		layout1->addWidget( activePropertyLabel, 1, 0 );
		layout1->addWidget( this->activePropertyComboBox, 1, 1 );

		QHBoxLayout * layout3 = new QHBoxLayout;
		layout3->addWidget( this->rangeLabel );
		layout3->addWidget( this->rangeSlider );
		layout3->addStretch();

		QGridLayout * layout5 = new QGridLayout;
		layout5->addWidget( histogramOpacityLabel, 0, 0, 1, 1 );
		layout5->addWidget( this->histogramOpacitySlider, 0, 1, 1, 3 );
		layout5->addWidget( this->rangeLabel, 1, 0, 1, 1 );
		layout5->addWidget( this->rangeSlider, 1, 1, 1, 3 );
		layout5->addWidget( this->selectColorButton, 2, 0, 1, 1 );
		layout5->addWidget( this->colorPatch, 2, 1, 1, 1 );
		layout5->addWidget( this->selectOutlineColorButton, 2, 2, 1, 1 );
		layout5->addWidget( this->outlineColorPatch, 2, 3, 1, 1 );
		layout5->addWidget( this->autoColorComboBox, 3, 0, 1, 2 );
		layout5->addWidget( this->selectBaseColorButton, 3, 2, 1, 1 );
		layout5->addWidget( this->baseColorPatch, 3, 3, 1, 1 );
		layout5->addWidget( this->applyStaircaseButton, 4, 0, 1, 2 );
		layout5->addWidget( this->applyInvertedStaircaseButton, 4, 2, 1, 2 );
		layout5->addWidget( this->applyEqualizeToSelectedButton, 5, 0, 1, 2 );
		layout5->addWidget( this->disableSelectedButton, 5, 2, 1, 2 );
		layout5->addWidget( this->subdivisionComboBox, 6, 0, 1, 2 );
		layout5->addWidget( this->numberOfIntervalsComboBox, 6, 2, 1, 2 );
		layout5->addWidget( this->savePropertiesButton, 7, 0, 1, 2 );
		layout5->addWidget( this->loadPropertiesButton, 7, 2, 1, 2 );
		layout5->addWidget( transformButton, 8, 0, 1, 2 );
		layout5->addWidget( screenshotButton, 8, 2, 1, 2 );
		layout5->addWidget( loadDATButton, 9, 0, 1, 2 );
		layout5->addWidget( startTimingButton, 9, 2, 1, 2 );

		QHBoxLayout * layout6 = new QHBoxLayout;
		layout6->addWidget( this->blurringCheckBox );
		layout6->addWidget( this->noiseCheckBox );
		layout6->addStretch();

		QVBoxLayout * layout = new QVBoxLayout;
		layout->addWidget( this->roiCheckBox );
		layout->addWidget( this->invertCheckBox );
		layout->addWidget( this->loadHeaderButton );
		layout->addLayout( layout1 );
		layout->addLayout( layout6 );
		layout->addWidget( this->histogramWidget );
		layout->addLayout( layout5 );
		layout->addStretch();

		this->initialized = true;

		this->widget = new QWidget;
		this->widget->setLayout( layout );
		return this->widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::loadTransform()
	{
		if( this->actor == 0 )
		{
			return;
		}

		QString fileName = QFileDialog::getOpenFileName( 0, "Open Transform" );
		if( fileName.isNull() || fileName.isEmpty() )
			return;

		QFile file(fileName);
		if( ! file.open( QIODevice::ReadOnly | QIODevice::Text) )
		{
			std::cout << "UncertaintyVisPlugin::loadTransform() " \
				"could not open transform " << fileName.toStdString() << std::endl;
			return;
		}

		float translate[3];
		float rotate[3];
		float scale[3];

		QString text(file.readAll());
		std::stringstream stream(text.toStdString());
		stream >> translate[0] >> translate[1] >> translate[2];
		stream >> rotate[0] >> rotate[1] >> rotate[2];
		stream >> scale[0] >> scale[1] >> scale[2];

		file.close();

		this->actor->SetPosition( translate[0], translate[1], translate[2] );
		this->actor->SetOrientation( rotate[0], rotate[1], rotate[2] );
		this->actor->SetScale( scale[0], scale[1], scale[2] );

		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::loadHeader()
	{
		QString fileName = QFileDialog::getOpenFileName(
				0, "Choose header file", ".", "*.*" );
		if( fileName.isNull() || fileName.isEmpty() )
			return;

		QFile file( fileName );
		file.open( QIODevice::ReadOnly | QIODevice::Text );
		if( file.error() )
		{
			std::cout << "ConfidenceVisPlugin::loadHeader() " \
					"could not open header file " << fileName.toStdString() << std::endl;
			return;
		}

		QFileInfo info( fileName );
		QTextStream stream( & file );
		QDir dir = info.dir();
		QString fileNameData  = dir.absolutePath() + "/" + stream.readLine();
		QString fileNameTable = dir.absolutePath() + "/" + stream.readLine();
		std::cout << "UncertaintyVisPlugin::loadHeader() " \
				"loaded polydata " << fileNameData.toStdString() << std::endl;
		std::cout << "UncertaintyVisPlugin::loadHeader() " \
				"loaded confidence table " << fileNameTable.toStdString() << std::endl;
		file.close();

		// Load polydata

		vtkPolyDataReader * reader = vtkPolyDataReader::New();
		reader->SetFileName( fileNameData.toStdString().c_str() );
		this->mapper->SetInput( reader->GetOutput() );
		this->mapper->Update();
		if( this->actor->GetMapper() == 0 )
			this->actor->SetMapper( this->mapper );
		reader->Delete();

		// Load confidence table

		QFile fileTable( fileNameTable );
		fileTable.open( QIODevice::ReadOnly | QIODevice::Text );
		if( fileTable.error() )
		{
			std::cout << "UncertaintyVisPlugin::loadHeader() " \
					"could not open confidence table " << fileNameTable.toStdString() << std::endl;
			return;
		}

		vtkConfidenceTable * table = vtkConfidenceTable::New();

		QTextStream streamTable( & fileTable );
		while( streamTable.atEnd() == false )
		{
			QString line = streamTable.readLine();
			QStringList tokens = line.split( " " );
			float confidence = tokens.at( 0 ).toFloat();
			int id = tokens.at( 1 ).toInt();
			table->Add( confidence, id );
		}

		fileTable.close();

		table->Normalize();
		if( this->invertCheckBox->isChecked() )
			table->Invert();
		this->mapper->SetTable( table );
		this->histogram->SetData(
				table->GetConfidenceValues(), table->GetNumberOfValues() );
		this->histogramWidget->setInterval( this->interval );
		this->histogramWidget->setHistogram( this->histogram );
		this->histogramWidget->reset();

		vtkMedicalCanvas * canvas = this->fullCore()->canvas();
		vtkRenderWindowInteractor * interactor = canvas->GetInteractor();
		this->roiWidget->GetRepresentation()->SetRenderer( canvas->GetRenderer3D() );
		this->roiWidget->SetInteractor( interactor );
		this->roiWidget->EnabledOn();

		canvas->GetRenderer3D()->ResetCamera();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::roiEnabled( bool enabled )
	{
		if( enabled )
		{
			this->roiWidget->ProcessEventsOn();
			this->mapper->SetROIEnabled( true );
		}
		else
		{
			this->mapper->SetROIEnabled( false );
			this->roiWidget->ProcessEventsOff();
		}

		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::blurringEnabled( bool enabled )
	{
		this->interval->GetProperties()->SetBlurringEnabled( enabled );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::noiseEnabled( bool enabled )
	{
		this->interval->GetProperties()->SetNoiseEnabled( enabled );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::activePropertyChanged( const QString name )
	{		
		if( name == "Opacity" )
		{
			this->interval->GetProperties()->SetActivePropertyToOpacity();
		}
		else if( name == "Outline opacity" )
		{
			this->interval->GetProperties()->SetActivePropertyToOutlineOpacity();
		}
		else if( name == "Outline thickness" )
		{
			this->interval->GetProperties()->SetActivePropertyToOutlineThickness();
		}
		else if( name == "Dilation" )
		{
			this->interval->GetProperties()->SetActivePropertyToDilation();
		}
		else if( name == "Checker size" )
		{
			this->interval->GetProperties()->SetActivePropertyToCheckerSize();
		}
		else if( name == "Hole size" )
		{
			this->interval->GetProperties()->SetActivePropertyToHoleSize();
		}
		else if( name == "Blurring radius" )
		{
			this->interval->GetProperties()->SetActivePropertyToBlurringRadius();
		}
		else if( name == "Blurring brightness" )
		{
			this->interval->GetProperties()->SetActivePropertyToBlurringBrightness();
		}
		else if( name == "Noise frequency" )
		{
			this->interval->GetProperties()->SetActivePropertyToNoiseFrequency();
		}
		else
		{
			std::cout << "UncertaintyVisPlugin::activePropertyChanged() " \
					"unknown property name " << name.toStdString() << std::endl;
		}

		this->histogramWidget->reset();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::renderModeChanged( const QString name )
	{
		if( name == "Solid" )
		{
			this->mapper->SetRenderModeToSolid();
		}
		else if( name == "Checker board" )
		{
			this->mapper->SetRenderModeToCheckerBoard();
		}
		else if( name == "Holes" )
		{
			this->mapper->SetRenderModeToHoles();
		}
		else
		{
			std::cout << "UncertaintyVisPlugin::renderModeChanged() " \
					"unknown render mode " << name.toStdString() << std::endl;
		}

		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::histogramOpacityChanged( int value )
	{
		this->histogramWidget->setOpacity( value / 100.0f );
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::intervalRangeChanged( int value )
	{
		float f = value / 100.0f;
		this->rangeLabel->setText( tr( "%1" ).arg( f ) );
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::intervalSliderReleased()
	{
		float value = this->rangeSlider->value() / 100.0f;
		this->interval->SetRange( value, 1.0f );
		this->interval->SetSubdivisionToEqualWidth();
		this->histogramWidget->reset();
		this->fullCore()->render();
		std::cout << "New interval minimum: " << value << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::intervalChanged()
	{
		// Check if the active property is noise frequency. If so, then
		// poke the mapper to update its noise textures
		
		this->subdivisionComboBox->setCurrentIndex( 2 );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::intervalSelected( int index )
	{
		this->selectedInterval = index;

		if( this->selectedInterval > -1 )
		{
			float * color = this->interval->GetProperties()->GetColor( this->selectedInterval );
			QColor newColor;
			newColor.setRedF( color[0] );
			newColor.setGreenF( color[1] );
			newColor.setBlueF( color[2] );
			QPalette colorPalette( this->colorPatch->palette() );
			colorPalette.setColor( QPalette::Background, newColor );
			this->colorPatch->setPalette( colorPalette );

			float * outlineColor = this->interval->GetProperties()->GetOutlineColor( this->selectedInterval );
			QColor newOutlineColor;
			newOutlineColor.setRedF( outlineColor[0] );
			newOutlineColor.setGreenF( outlineColor[1] );
			newOutlineColor.setBlueF( outlineColor[2] );
			QPalette outlineColorPalette( this->colorPatch->palette() );
			outlineColorPalette.setColor( QPalette::Background, newOutlineColor );
			this->outlineColorPatch->setPalette( outlineColorPalette );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::colorSelected()
	{
		if( this->selectedInterval == -1 )
		{
			return;
		}

		QColor current = this->colorPatch->palette().color( QPalette::Background );
		QColor color = QColorDialog::getColor( current, 0 );
		if(	color.isValid() == false || color == current )
		{
			return;
		}

		this->interval->GetProperties()->SetColor(
				this->selectedInterval, color.redF(), color.greenF(), color.blueF() );

		QPalette p( this->colorPatch->palette() );
		p.setColor( QPalette::Background, color );
		this->colorPatch->setPalette( p );

		// User has selected specific color so the auto-coloring scheme
		// is now 'Custom'

		this->autoColorComboBox->setCurrentIndex( 0 );
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::outlineColorSelected()
	{
		if( this->selectedInterval == -1 )
		{
			return;
		}

		QColor current = this->outlineColorPatch->palette().color( QPalette::Background );
		QColor color = QColorDialog::getColor( current, 0 );
		if(	color.isValid() == false || color == current )
		{
			return;
		}

		this->interval->GetProperties()->SetOutlineColor(
				this->selectedInterval, color.redF(), color.greenF(), color.blueF() );

		QPalette p( this->outlineColorPatch->palette() );
		p.setColor( QPalette::Background, color );
		this->outlineColorPatch->setPalette( p );

		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::baseColorSelected()
	{
		QColor current = this->baseColorPatch->palette().color( QPalette::Background );
		QColor color = QColorDialog::getColor( current, 0 );
		if(	color.isValid() == false || color == current )
		{
			return;
		}

		QPalette p( this->baseColorPatch->palette() );
		p.setColor( QPalette::Background, color );
		this->baseColorPatch->setPalette( p );
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::autoColorSchemeSelected( const QString name )
	{
		int nrIntervals = this->interval->GetNumberOfIntervals();
		QColor baseColor = this->baseColorPatch->palette().color( QPalette::Background );

		float delta[3];
		delta[0] = baseColor.redF() / nrIntervals;
		delta[1] = baseColor.greenF() / nrIntervals;
		delta[2] = baseColor.blueF() / nrIntervals;

		if( name == "Custom" )
		{
		}
		else if( name == "Light to dark" )
		{
			float color[3];
			color[0] = baseColor.redF();
			color[1] = baseColor.greenF();
			color[2] = baseColor.blueF();

			for( int i = 0; i < nrIntervals; ++i )
			{
				this->interval->GetProperties()->SetColor( 
					i, color[0], color[1], color[2] );
				color[0] -= delta[0];
				color[1] -= delta[1];
				color[2] -= delta[2];
			}
		}
		else if( name == "Dark to light" )
		{
			float color[3];
			color[0] = delta[0];
			color[1] = delta[1];
			color[2] = delta[2];

			for( int i = 0; i < nrIntervals; ++i )
			{
				this->interval->GetProperties()->SetColor( 
					i, color[0], color[1], color[2] );
				color[0] += delta[0];
				color[1] += delta[1];
				color[2] += delta[2];
			}
		}
		else if( name == "Decreasing saturation" )
		{
			qreal h, s, v;
			baseColor.getHsvF( & h, & s, & v );
			float delta = s / nrIntervals;
			float saturation = s;

			for( int i = 0; i < nrIntervals; ++i )
			{
				QColor newColor;
				newColor.setHsvF( h, saturation, v );
				this->interval->GetProperties()->SetColor( 
					i, newColor.redF(), newColor.greenF(), newColor.blueF() );
				saturation -= delta;
			}
		}
		else if( name == "Increasing saturation" )
		{
			qreal h, s, v;
			baseColor.getHsvF( & h, & s, & v );
			float delta = s / nrIntervals;
			float saturation = delta;

			for( int i = 0; i < nrIntervals; ++i )
			{
				QColor newColor;
				newColor.setHsvF( h, saturation, v );
				this->interval->GetProperties()->SetColor( 
					i, newColor.redF(), newColor.greenF(), newColor.blueF() );
				saturation += delta;
			}
		}
		else {}

		this->histogramWidget->reset();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::staircaseApplied()
	{
		vtkConfidenceInterval * ci = this->interval;
		if( ci->GetNumberOfIntervals() > 0 )
		{
			float increment = 1.0f / ci->GetNumberOfIntervals();
			float value = increment;

			for( int i = 0; i < ci->GetNumberOfIntervals(); ++i )
			{
				ci->GetProperties()->SetValue( i, value );
				value += increment;
			}

			this->histogramWidget->reset();
			this->fullCore()->render();
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::invertedStaircaseApplied()
	{
		vtkConfidenceInterval * ci = this->interval;
		if( ci->GetNumberOfIntervals() > 0 )
		{
			float decrement = 1.0f / ci->GetNumberOfIntervals();
			float value = 1.0f;
			for( int i = 0; i < ci->GetNumberOfIntervals(); ++i )
			{
				ci->GetProperties()->SetValue( i, value );
				value -= decrement;
			}

			this->histogramWidget->reset();
			this->fullCore()->render();
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::equalizeToSelectedApplied()
	{
		if( this->selectedInterval < 0 )
			return;

		vtkConfidenceInterval * ci = this->interval;
		if( ci->GetNumberOfIntervals() > 0 )
		{
			float value = ci->GetProperties()->GetValue( this->selectedInterval );
			for( int i = 0; i < ci->GetNumberOfIntervals(); ++i )
			{
				ci->GetProperties()->SetValue( i, value );
			}
		}

		this->histogramWidget->reset();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::disableSelected()
	{
		if( this->selectedInterval < 0 )
			return;

		vtkConfidenceInterval * ci = this->interval;
		if( ci->GetNumberOfIntervals() > 0 )
		{
			bool enabled = ci->GetProperties()->IsEnabled( this->selectedInterval );
			enabled = ! enabled;
			ci->GetProperties()->SetEnabled( this->selectedInterval, enabled );
		}

		this->histogramWidget->reset();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::subdivisionChanged( const QString name )
	{
		if( name == "Equal width" )
		{
			this->interval->SetSubdivisionToEqualWidth();
		}
		else if( name == "Equal histogram area" )
		{
			this->interval->SetSubdivisionToEqualHistogramArea();
		}
		else 
		{
			this->interval->SetSubdivisionToCustomWidth();
		}

		this->histogramWidget->reset();
		this->fullCore()->render();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::numberOfIntervalsChanged( int index )
	{
		int number = index + 2;
		if( number != this->interval->GetNumberOfIntervals() )
		{
			this->interval->SetNumberOfIntervals( number );
			this->interval->SetSubdivisionToEqualWidth();
			this->histogramWidget->reset();
			this->fullCore()->render();
		}
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::saveProperties()
	{
		QString fileName = QFileDialog::getSaveFileName( 0, "Save properties" );
		if( fileName.isNull() || fileName.isEmpty() )
			return;

		QFile file( fileName );
		file.open( QIODevice::WriteOnly | QIODevice::Text );
		if( file.error() )
		{
			std::cout << "ConfidenceVisPlugin::saveProperties() " \
					"could not open file " << fileName.toStdString() << std::endl;
			return;
		}

		file.write( this->interval->WriteToString().c_str() );
		file.close();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::loadProperties()
	{
		QString fileName = QFileDialog::getOpenFileName( 0, "Load properties" );
		if( fileName.isNull() || fileName.isEmpty() )
			return;

		QFile file( fileName );
		file.open( QIODevice::ReadOnly | QIODevice::Text );
		if( file.error() )
		{
			std::cout << "ConfidenceVisPlugin::loadProperties() " \
					"could not open file " << fileName.toStdString() << std::endl;
			return;
		}

		QString settings( file.readAll() );
		this->interval->ReadFromString( settings.toStdString() );
		this->histogramWidget->reset();
		this->fullCore()->render();
		file.close();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::makeScreenshot()
	{
		//this->fullCore()->gui()->makeScreenshot();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::loadDAT()
	{
		//vtkDTIComponentReader * reader = vtkDTIComponentReader::New();
		//reader->SetFileName( "M:/Datasets/EuroVis2011/Kempenhaeghe/mm10.dtiI.dat" );
		//reader->SetDataScalarTypeToFloat();
		//reader->SetDataSpacing( 2, 2, 2 );
		//reader->Update();

		vtkImageReader2 * reader = vtkImageReader2::New();
		reader->SetFileName( "M:/Datasets/EuroVis2011/Kempenhaeghe/T2_volume.raw" );
		reader->SetFileDimensionality( 3 );
		reader->SetDataScalarTypeToFloat();
		reader->SetDataSpacing( 2, 2, 2 );
		reader->SetDataExtent( 0, 80, 0, 105, 0, 75 );
		reader->Update();

		data::DataSet * dataset = new data::DataSet("T2_volume.raw","scalar volume",reader->GetOutput());
		this->fullCore()->data()->addDataSet( dataset );

		//vtkImageWriter * writer = vtkImageWriter::New();
		//writer->SetFileName( "M:/Datasets/EuroVis2011/Kempenhaeghe/T2_volume.raw" );
		//writer->SetFileDimensionality( 3 );
		//writer->SetInput( reader->GetOutput() );
		//writer->Update();
		//writer->Write();

		//reader->Delete();
		//writer->Delete();
	}

	///////////////////////////////////////////////////////////////////////////
	void UncertaintyVisPlugin::startTiming()
	{
		double elapsed = 0.0;
		for( int i = 0; i < 10; ++i )
		{
			this->fullCore()->render();
			elapsed += this->fullCore()->canvas()->GetRenderer3D()->GetLastRenderTimeInSeconds();
			std::cout << elapsed << std::endl;
		}

		if( elapsed > 0.0 )
		std::cout << "Average FPS: " << 10.0 / elapsed << std::endl;
	}
}

Q_EXPORT_PLUGIN2( libUncertaintyVisPlugin, bmia::UncertaintyVisPlugin )