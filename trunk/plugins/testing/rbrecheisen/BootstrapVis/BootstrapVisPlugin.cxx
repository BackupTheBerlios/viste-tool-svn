#include "BootstrapVisPlugin.h"

#include "vtkPolyData.h"
#include "vtkAssembly.h"
#include "vtkActor.h"
#include "vtkTubeFilter.h"
#include "vtkProperty.h"

#include "vtkFiberTubeMapper.h"
#include "vtkFiberConfidenceMapper.h"
#include "vtkDistanceTable.h"
#include "vtkColor4.h"

#include <QVBoxLayout>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QCheckBox>
#include <QSlider>
#include <QLineEdit>
#include <QColorDialog>
#include <QString>

#include <vector>
#include <string>
#include <sstream>

namespace bmia
{
	////////////////////////////////////////////////////////////////////////
	BootstrapVisPlugin::BootstrapVisPlugin() : Plugin( "Bootstrap" )
	{
		this->selectedLevel = 0;
		this->levels = new std::vector< std::pair< std::string, float > >;
		this->fillColors = new std::vector< std::pair< float, vtkColor4 > >;
		this->lineColors = new std::vector< std::pair< float, vtkColor4 > >;

		// VTK

		this->mainMapper = vtkFiberTubeMapper::New();
		this->mainActor = vtkActor::New();
		this->mainActor->VisibilityOff();
		this->mainActor->GetProperty()->SetDiffuse( 0.6 );
		this->mainActor->GetProperty()->SetSpecular( 1.0 );
		this->mainActor->GetProperty()->SetSpecularPower( 64.0 );
		this->mainActor->GetProperty()->SetSpecularColor( 1.0, 1.0, 1.0 );
		this->mainActor->GetProperty()->SetColor( 0.0, 0.0, 1.0 );
		this->mainActor->SetMapper( this->mainMapper );

		this->bootstrapMapper = vtkFiberConfidenceMapper::New();
		this->bootstrapMapper->SetFillDilation( 5 );
		this->bootstrapMapper->SetOutlineWidth( 3 );
		this->bootstrapMapper->SetDepthThreshold( 5.0f );
		this->bootstrapMapper->SetMaximumFiberDensity( 10 );
		this->bootstrapMapper->SetDensityColor( 1.0f, 1.0f, 1.0f, 1.0f );
		this->bootstrapMapper->SetDensityColoringEnabled( false );
		this->bootstrapMapper->SetDensitySmoothingEnabled( false );
		this->bootstrapMapper->SetDensityWeightingEnabled( false );
		this->bootstrapMapper->SetErosionEnabled( false );

		this->bootstrapActor = vtkActor::New();
		this->bootstrapActor->VisibilityOff();
		this->bootstrapActor->SetMapper( this->bootstrapMapper );

		this->actor = vtkAssembly::New();
		this->actor->AddPart( this->mainActor );
		this->actor->AddPart( this->bootstrapActor );

		this->table = 0;
		this->mainFibers = 0;
		this->mainFibersPtr = 0;
		this->bootstrapFibers = 0;
		this->tubeFilter = 0;

		// WIDGET

		QLabel * label1 = new QLabel( "Dataset" );
		this->dataBox = new QComboBox;

		QPushButton * button1 = new QPushButton( "Load table" );
		this->connect( button1, SIGNAL( clicked() ), this, SLOT( loadTable() ) );

		QCheckBox * check1 = new QCheckBox( "Show main fibers" );
		check1->setChecked( true );
		this->connect( check1, SIGNAL( stateChanged( int ) ), this, SLOT( showMainFibers( int ) ) );

		QCheckBox * check2 = new QCheckBox( "Show confidence fibers" );
		check2->setChecked( true );
		this->connect( check2, SIGNAL( stateChanged( int ) ), this, SLOT( showConfidenceFibers( int ) ) );

		QCheckBox * check3 = new QCheckBox( "Show main fibers as tubes" );
		check3->setChecked( false );
		this->connect( check3, SIGNAL( stateChanged( int ) ), this, SLOT( showMainFibersAsStreamtubes( int ) ) );

		QCheckBox * check4 = new QCheckBox( "Enable density coloring" );
		check4->setChecked( false );
		this->connect( check4, SIGNAL( stateChanged( int ) ), this, SLOT( enableDensityColoring( int ) ) );

		QCheckBox * check5 = new QCheckBox( "Enable density smoothing" );
		check5->setChecked( false );
		this->connect( check5, SIGNAL( stateChanged( int ) ), this, SLOT( enableDensitySmoothing( int ) ) );

		QCheckBox * check6 = new QCheckBox( "Enable density weighting" );
		check6->setChecked( false );
		this->connect( check6, SIGNAL( stateChanged( int ) ), this, SLOT( enableDensityWeighting( int ) ) );

		QCheckBox * check7 = new QCheckBox( "Enable erosion" );
		check7->setChecked( false );
		this->connect( check7, SIGNAL( stateChanged( int ) ), this, SLOT( enableErosion( int ) ) );

		QLabel * label2 = new QLabel( "Tube radius" );
		QLineEdit * edit1 = new QLineEdit( "0.5" );
		QHBoxLayout * layout1 = new QHBoxLayout;
		layout1->addWidget( label2 );
		layout1->addWidget( edit1 );
		layout1->addStretch();

		QPushButton * button2 = new QPushButton( "Main fiber color" );
		this->connect( button2, SIGNAL( clicked() ), this, SLOT( selectMainFiberColor() ) );

		QLabel * label3 = new QLabel( "Confidence levels" );
		this->levelBox = new QComboBox;
		this->levelBox->addItem( "95%" );
		this->levelBox->addItem( "50%" );
		this->levelBox->addItem( "10%" );
		this->levelBox->setCurrentIndex( 0 );
		this->connect( this->levelBox, SIGNAL( currentIndexChanged( int ) ), this, SLOT( levelSelected( int ) ) );
		QHBoxLayout * layout2 = new QHBoxLayout;
		layout2->addWidget( label3 );
		layout2->addWidget( this->levelBox );
		layout2->addStretch();

		this->slider = new QSlider( Qt::Horizontal );
		this->slider->setRange( 0, 100 );
		this->slider->setValue( 95 );
		this->sliderLabel = new QLabel( "95%" );
		this->connect( this->slider, SIGNAL( valueChanged( int ) ), this, SLOT( levelChanged( int ) ) );
		QHBoxLayout * layout3 = new QHBoxLayout;
		layout3->addWidget( this->sliderLabel );
		layout3->addWidget( this->slider );
		layout3->addStretch();

		QPushButton * button5 = new QPushButton( "Select fill color" );
		this->connect( button5, SIGNAL( clicked() ), this, SLOT( selectFillColor() ) );

		QPushButton * button6 = new QPushButton( "Select line color" );
		this->connect( button6, SIGNAL( clicked() ), this, SLOT( selectLineColor() ) );

		QPushButton * button7 = new QPushButton( "Update" );
		this->connect( button7, SIGNAL( clicked() ), this, SLOT( updateFibers() ) );

		QPushButton * button8 = new QPushButton( "Show as main fibers" );
		this->connect( button8, SIGNAL( clicked() ), this, SLOT( showAsMainFibers() ) );

		QPushButton * button9 = new QPushButton( "Show as bootstrap fibers" );
		this->connect( button9, SIGNAL( clicked() ), this, SLOT( showAsBootstrapFibers() ) );

		QPushButton * button10 = new QPushButton( "Select density color" );
		this->connect( button10, SIGNAL( clicked() ), this, SLOT( selectDensityColor() ) );

		QVBoxLayout * layout = new QVBoxLayout;
		layout->addWidget( label1 );
		layout->addWidget( this->dataBox );
		layout->addWidget( button1 );
		layout->addWidget( check1 );
		layout->addWidget( check2 );
		layout->addWidget( check3 );
		layout->addWidget( check4 );
		layout->addWidget( check5 );
		layout->addWidget( check6 );
		layout->addWidget( check7 );
		layout->addLayout( layout1 );
		layout->addWidget( button2 );
		layout->addLayout( layout2 );
		layout->addLayout( layout3 );
		layout->addWidget( button5 );
		layout->addWidget( button6 );
		layout->addWidget( button8 );
		layout->addWidget( button9 );
		layout->addWidget( button10 );
		layout->addWidget( button7 );
		layout->addStretch();

		this->widget = new QWidget;
		this->widget->setLayout( layout );
	}

	////////////////////////////////////////////////////////////////////////
	BootstrapVisPlugin::~BootstrapVisPlugin()
	{
		this->mainMapper->Delete();
		this->mainActor->Delete();
		this->bootstrapMapper->Delete();
		this->bootstrapActor->Delete();
		this->actor->Delete();

		delete this->dataBox;
		delete this->levelBox;
		delete this->widget;
	}

	////////////////////////////////////////////////////////////////////////
	vtkProp * BootstrapVisPlugin::getVtkProp()
	{
		return this->actor;
	}

	////////////////////////////////////////////////////////////////////////
	QWidget * BootstrapVisPlugin::getGUI()
	{
		return this->widget;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::dataSetAdded( data::DataSet * dataset )
	{
		if( dataset->getKind() == "polydata" )
		{
			QString name = dataset->getName();
			QMap< QString, data::DataSet * >::iterator iter = this->datasets.find( name );
			if( iter != this->datasets.end() )
			{
				std::cout << "BootstrapVisPlugin::dataSetAdded() dataset already loaded" << std::endl;
				return;
			}

			this->datasets[name] = dataset;
			this->dataBox->addItem( name );
			this->dataBox->setCurrentIndex( this->dataBox->count() - 1 );
		}
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::dataSetChanged(data::DataSet * _dataset )
	{
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::dataSetRemoved(data::DataSet * _dataset )
	{
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::loadTable()
	{
		QString fileName = QFileDialog::getOpenFileName( 0, "Open distance table", "." );
		if( fileName.isNull() || fileName.isEmpty() )
			return;

		if( this->table != 0 )
			this->table->Delete();
		this->table = vtkDistanceTable::New();
		this->table->Register( 0 );

		std::ifstream file( fileName.toStdString().c_str(), std::ios::in );
		if( file.fail() )
		{
			std::cout << "BootstrapVisPlugin::loadTable() could not open table" << std::endl;
			return;
		}

		double distance;
		int offset;

		while( file.eof() == false )
		{
			file >> distance >> offset;
			this->table->Add( distance, offset );
		}

		file.close();

		this->bootstrapMapper->SetDistanceTable( this->table );

		if( this->levels == 0 )
			this->levels = new std::vector< std::pair< std::string, float > >;
		this->levels->push_back( std::pair< std::string, float >( "95%", 0.95f ) );
		this->levels->push_back( std::pair< std::string, float >( "50%", 0.5f ));
		this->levels->push_back( std::pair< std::string, float >( "10%", 0.1f ));

		if( this->fillColors == 0 )
			this->fillColors = new std::vector< std::pair< float, vtkColor4 > >;
		this->fillColors->push_back( std::pair< float, vtkColor4 >( 0.95f, vtkColor4( 64, 64, 64, 255 ) ) );
		this->fillColors->push_back( std::pair< float, vtkColor4 >( 0.5f, vtkColor4( 128, 128, 128, 255 ) ) );
		this->fillColors->push_back( std::pair< float, vtkColor4 >( 0.1f, vtkColor4( 255, 255, 255, 255 ) ) );

		if( this->lineColors == 0 )
			this->lineColors = new std::vector< std::pair< float, vtkColor4 > >;
		this->lineColors->push_back( std::pair< float, vtkColor4 >( 0.95f, vtkColor4( 255, 255, 255, 255 ) ) );
		this->lineColors->push_back( std::pair< float, vtkColor4 >( 0.5f, vtkColor4( 255, 255, 255, 255 ) ) );
		this->lineColors->push_back( std::pair< float, vtkColor4 >( 0.1f, vtkColor4( 255, 255, 255, 255 ) ) );

		this->bootstrapMapper->SetConfidenceLevels( this->levels );
		this->bootstrapMapper->SetFillColors( this->fillColors );
		this->bootstrapMapper->SetLineColors( this->lineColors );

		if( this->bootstrapMapper->GetInput() != 0 )
		{
			this->bootstrapMapper->Update();
			this->bootstrapActor->VisibilityOn();
		}

		this->selectedLevel = 0;
		this->levelBox->setCurrentIndex( 0 );

		this->core()->render();

		std::cout << "BootstrapVisPlugin::loadTable() added distance table" << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::showMainFibers( int _state )
	{
		if( this->mainActor == 0 )
			return;
		this->mainActor->SetVisibility( _state == Qt::Checked ? 1 : 0 );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::showConfidenceFibers( int _state )
	{
		if( this->bootstrapActor == 0 )
			return;
		this->bootstrapActor->SetVisibility( _state == Qt::Checked ? 1 : 0 );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::showMainFibersAsStreamtubes( int _state )
	{
		if( this->mainActor == 0 || this->mainFibers == 0 )
			return;

		this->mainFibersPtr = this->mainFibers;
		this->mainMapper->SetShadersEnabled( false );

		bool showTubes = (_state == Qt::Checked) ? true : false;
		if( showTubes )
		{
			if( this->tubeFilter == 0 )
				this->tubeFilter = vtkTubeFilter::New();
			this->tubeFilter->SetInput( this->mainFibers );
			this->tubeFilter->SetRadius( 0.4 );
			this->tubeFilter->SetNumberOfSides( 16 );
			this->tubeFilter->CappingOn();
			this->tubeFilter->Update();

			this->mainFibersPtr = this->tubeFilter->GetOutput();
			this->mainMapper->SetShadersEnabled( true );
		}

		this->mainMapper->SetInput( this->mainFibersPtr );
		this->mainMapper->Update();

		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::enableDensityColoring( int _state )
	{
		bool enabled = (_state == Qt::Checked) ? true : false;
		this->bootstrapMapper->SetDensityColoringEnabled( enabled );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::enableDensitySmoothing( int _state )
	{
		bool enabled = (_state == Qt::Checked) ? true : false;
		this->bootstrapMapper->SetDensitySmoothingEnabled( enabled );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::enableErosion( int _state )
	{
		bool enabled = (_state == Qt::Checked) ? true : false;
		this->bootstrapMapper->SetErosionEnabled( enabled );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::enableDensityWeighting( int _state )
	{
		bool enabled = (_state == Qt::Checked) ? true : false;
		this->bootstrapMapper->SetDensityWeightingEnabled( enabled );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::selectMainFiberColor()
	{		
		QColor color = QColorDialog::getColor();
		if( color.isValid() == false )
			return;
		this->mainActor->GetProperty()->SetColor( color.redF(), color.greenF(), color.blueF() );
		this->core()->render();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::selectFillColor()
	{
		QColor color = QColorDialog::getColor();
		if( color.isValid() == false )
			return;

		int i = this->selectedLevel;
		this->fillColors->at( i ).second.r = color.red();
		this->fillColors->at( i ).second.g = color.green();
		this->fillColors->at( i ).second.b = color.blue();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::selectLineColor()
	{
		QColor color = QColorDialog::getColor();
		if( color.isValid() == false )
			return;

		int i = this->selectedLevel;
		this->lineColors->at( i ).second.r = color.red();
		this->lineColors->at( i ).second.g = color.green();
		this->lineColors->at( i ).second.b = color.blue();
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::selectDensityColor()
	{
		QColor color = QColorDialog::getColor();
		if( color.isValid() == false )
			return;

		this->bootstrapMapper->SetDensityColor(
				color.redF(), color.greenF(), color.blueF(), 1.0f );
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::levelChanged( int _value )
	{
		QString text = QString( "%1\%" ).arg( _value );
		this->sliderLabel->setText( text );
		this->levelBox->setItemText( this->levelBox->currentIndex(), text );

		float f = _value / 100.0f;
		int i = this->selectedLevel;
		this->levels->at( i ).first = text.toStdString();
		this->levels->at( i ).second = f;
		this->fillColors->at( i ).first = f;
		this->lineColors->at( i ).first = f;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::levelSelected( int _index )
	{
		QString text = this->levelBox->currentText();
		text.chop( 1 );
		int value = text.toInt();
		this->slider->setValue( value );

		float fValue = value / 100.0f;
		for( int i = 0; i < 3; ++i )
			if( this->levels->at( i ).second == fValue )
				this->selectedLevel = i;
		std::cout << "levelSelected() " << this->selectedLevel << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::showAsMainFibers()
	{
		QString name = this->dataBox->currentText();
		data::DataSet * dataset = this->datasets[name];

		this->mainFibers = dataset->getVtkPolyData();
		this->mainMapper->SetInput( this->mainFibers );
		this->mainMapper->Update();
		this->mainActor->VisibilityOn();

		this->core()->render();

		std::cout << "BootstrapVisPlugin::showAsMainFibers()" << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::showAsBootstrapFibers()
	{
		QString name = this->dataBox->currentText();
		data::DataSet * dataset = this->datasets[name];

		this->bootstrapFibers = dataset->getVtkPolyData();
		this->bootstrapMapper->SetInput( this->bootstrapFibers );

		if( this->table != 0 )
		{
			this->bootstrapMapper->SetDistanceTable( this->table );
			this->bootstrapActor->VisibilityOn();
		}

		this->bootstrapMapper->Update();

		this->core()->render();

		std::cout << "BootstrapVisPlugin::showAsBootstrapFibers()" << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	void BootstrapVisPlugin::updateFibers()
	{
		if( this->levels->size() == 0 )
			return;

		this->bootstrapMapper->SetConfidenceLevels( this->levels );
		this->bootstrapMapper->SetFillColors( this->fillColors );
		this->bootstrapMapper->SetLineColors( this->lineColors );
		this->bootstrapMapper->Update();

		this->core()->render();
	}
}

Q_EXPORT_PLUGIN2( BootstrapVisPlugin, bmia::BootstrapVisPlugin )
