// Includes DTI tool
#include <QDistanceWidget.h>

// Includes C++
#include <iostream>

///////////////////////////////////////////////////////////////////////////
QDistanceWidget::QDistanceWidget( QWidget * parent ) : QTabWidget( parent )
{
	_mapWidget = new QTumorMapWidget();
	_graphWidget = new QMinimalDistanceWidget();

	_button = new QPushButton( "Compute" );
	this->connect( _button, SIGNAL( clicked() ), this, SLOT( compute() ) );

    _buttonComputeDT = new QPushButton( "Compute Single DT" );
    this->connect( _buttonComputeDT, SIGNAL(clicked()), this, SLOT(computeSingleDT()) );

	_riskLabel = new QLabel( "0 mm" );
    _probabilityLabel = new QLabel( "+/- 0 %" );

	_riskSlider = new QSlider( Qt::Horizontal );
	_riskSlider->setRange( 0, 100 );
	_riskSlider->setValue( 0 );
	this->connect( _riskSlider, SIGNAL( valueChanged( int ) ), this, SLOT( riskChanged( int ) ) );

	_speedSlider = new QSlider( Qt::Horizontal );
	_speedSlider->setRange( 32, 128 );
	_speedSlider->setValue( 32 );

    _probabilitySlider = new QSlider( Qt::Horizontal );
    _probabilitySlider->setRange( 0, 50 );
    _probabilitySlider->setValue( 0 );
    this->connect( _probabilitySlider, SIGNAL(valueChanged(int)), this, SLOT(probabilityChanged(int)) );

	_riskEnabled = new QCheckBox( "Risk Enabled" );
	_riskEnabled->setChecked( false );
	this->connect( _riskEnabled, SIGNAL( toggled( bool ) ), this, SLOT( riskEnabled( bool ) ) );

    _contoursEnabled = new QCheckBox( "Contours Enabled" );
	_contoursEnabled->setChecked( false );
	this->connect( _contoursEnabled, SIGNAL( toggled( bool ) ), this, SLOT( contoursEnabled( bool ) ) );

	_projectionEnabled = new QCheckBox( "Projection Enabled" );
	_projectionEnabled->setChecked( false );
	this->connect( _projectionEnabled, SIGNAL( toggled( bool ) ), this, SLOT( projectionEnabled( bool ) ) );

    _uncertaintyEnabled = new QCheckBox( "Uncertainty Enabled" );
    _uncertaintyEnabled->setChecked( false );
    this->connect( _uncertaintyEnabled, SIGNAL( toggled( bool ) ), this, SLOT( uncertaintyEnabled( bool ) ) );

    _automaticViewPointsEnabled = new QCheckBox( "View-Point Animation Enabled" );
	_automaticViewPointsEnabled->setChecked( false );
	this->connect( _automaticViewPointsEnabled, SIGNAL( toggled( bool ) ), this, SLOT( viewPointsEnabled( bool ) ) );

	_updateIsoValue = new QCheckBox( "Update Iso-Value" );
	_updateIsoValue->setChecked( false );
	this->connect( _updateIsoValue, SIGNAL( toggled( bool ) ), this, SLOT( updateIsoValue( bool ) ) );

    _invertDistance = new QCheckBox( "Invert Distances" );
    _invertDistance->setChecked( false );

	_contoursEdit = new QLineEdit( "0" );

	_fiberBox = new QComboBox;
	_tumorBox = new QComboBox;
    _volumeBox = new QComboBox;
	_tumorMeshBox = new QComboBox;

	_colorLookupBox = new QComboBox;
	_colorLookupBox->addItem( "Grayscale" );
	_colorLookupBox->addItem( "HeatMap" );
    _colorLookupBox->addItem( "CoolToWarm" );
    this->connect(_colorLookupBox, SIGNAL(currentIndexChanged(int)), this, SLOT(colorLookupChanged(int)));

	_configurationBox = new QComboBox;
	this->connect( _configurationBox, SIGNAL( currentIndexChanged( const QString ) ),
				  this, SLOT( configChanged( const QString ) ) );

	_samplesEdit = new QLineEdit( "0" );
	_rowsEdit = new QLineEdit( "0" );
	_columnsEdit = new QLineEdit( "0" );

	QHBoxLayout * layout2 = new QHBoxLayout;
	layout2->setContentsMargins( 1, 1, 1, 1 );
	layout2->addWidget( new QLabel( "Rows" ) );
	layout2->addWidget( _rowsEdit );
	layout2->addWidget( new QLabel( "Columns" ) );
	layout2->addWidget( _columnsEdit );
	layout2->addStretch();

	QVBoxLayout * layout = new QVBoxLayout;
	layout->setContentsMargins( 1, 1, 1, 1 );
	layout->addWidget( new QLabel( "Nr. Samples" ) );
	layout->addWidget( _samplesEdit );
	layout->addLayout( layout2 );
	layout->addWidget( new QLabel( "Fibers" ) );
	layout->addWidget( _fiberBox );
	layout->addWidget( new QLabel( "Tumor" ) );
	layout->addWidget( _tumorBox );
	layout->addWidget( new QLabel( "TumorMesh" ) );
	layout->addWidget( _tumorMeshBox );
    layout->addWidget( new QLabel( "Volume" ) );
    layout->addWidget( _volumeBox );
	layout->addWidget( _button );
    layout->addWidget( _invertDistance );
    layout->addWidget( _buttonComputeDT );
	layout->addStretch();

	QWidget * tab1 = new QWidget;
	tab1->setLayout( layout );

	QHBoxLayout * layout3 = new QHBoxLayout;
	layout3->addWidget( _riskSlider );
	layout3->addWidget( _riskLabel );
	layout3->addStretch();

	QHBoxLayout * layout4 = new QHBoxLayout;
	layout4->addWidget( _contoursEnabled );
	layout4->addWidget( _contoursEdit );
	layout4->addStretch();

	QHBoxLayout * layout5 = new QHBoxLayout;
	layout5->addWidget( new QLabel( "Fast" ) );
	layout5->addWidget( _speedSlider );
	layout5->addWidget( new QLabel( "Slow" ) );
	layout5->addStretch();

    QHBoxLayout * layout6 = new QHBoxLayout;
    layout6->addWidget( _probabilitySlider );
    layout6->addWidget( _probabilityLabel );
    layout6->addStretch();

	QVBoxLayout * layout1 = new QVBoxLayout;
	layout1->setContentsMargins( 1, 1, 1, 1 );
	layout1->addWidget( new QLabel( "Distance Uncertainty Graph" ) );
	layout1->addWidget( _graphWidget );
	layout1->addWidget( new QLabel( "Tumor Distance Map" ) );
	layout1->addWidget( _mapWidget );
	layout1->addLayout( layout4 );
	layout1->addWidget( _riskEnabled );
	layout1->addLayout( layout3 );
	layout1->addWidget( _updateIsoValue );
	layout1->addWidget( _projectionEnabled );
    layout1->addWidget( _uncertaintyEnabled );
    layout1->addLayout( layout6 );
    layout1->addWidget( _automaticViewPointsEnabled );
	layout1->addLayout( layout5 );
	layout1->addWidget( new QLabel( "Color Tables" ) );
	layout1->addWidget( _colorLookupBox );
	layout1->addWidget( new QLabel( "Configurations" ) );
	layout1->addWidget( _configurationBox );
	layout1->addStretch();

	QWidget * tab2 = new QWidget;
	tab2->setLayout( layout1 );

	this->addTab( tab1, "Datasets" );
	this->addTab( tab2, "Graphs" );
}

///////////////////////////////////////////////////////////////////////////
QDistanceWidget::~QDistanceWidget()
{
	delete _graphWidget;
	delete _mapWidget;
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::addDataset( const QString name )
{
	_fiberBox->addItem( name );
	_tumorBox->addItem( name );
    _volumeBox->addItem( name );

	if( name.contains( "tumor" ) )
		_tumorBox->setCurrentIndex( _tumorBox->count() - 1 );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::addPolyDataset( const QString name )
{
	_tumorMeshBox->addItem( name );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::addConfiguration( const QString name )
{
	_configurationBox->blockSignals( true );
	_configurationBox->addItem( name );
	_configurationBox->setCurrentIndex( _configurationBox->count() - 1 );
	_configurationBox->blockSignals( false );
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::selectedFiberIndex()
{
	return _fiberBox->currentIndex();
}

///////////////////////////////////////////////////////////////////////////
QString QDistanceWidget::selectedFiberName()
{
	return _fiberBox->currentText();
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::selectedTumorIndex()
{
	return _tumorBox->currentIndex();
}

///////////////////////////////////////////////////////////////////////////
QString QDistanceWidget::selectedTumorName()
{
	return _tumorBox->currentText();
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::selectedTumorMeshIndex()
{
	return _tumorMeshBox->currentIndex();
}

///////////////////////////////////////////////////////////////////////////
QString QDistanceWidget::selectedTumorMeshName()
{
	return _tumorMeshBox->currentText();
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::selectedVolumeIndex()
{
    return _volumeBox->currentIndex();
}

///////////////////////////////////////////////////////////////////////////
QString QDistanceWidget::selectedVolumeName()
{
    return _volumeBox->currentText();
}

///////////////////////////////////////////////////////////////////////////
QString QDistanceWidget::selectedColorLookupTableName()
{
	return _colorLookupBox->currentText();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setNumberOfSamples( int number )
{
	_samplesEdit->setText( QString( "%1" ).arg( number ) );
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::numberOfSamples()
{
	return _samplesEdit->text().toInt();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setNumberOfRows( int rows )
{
	_rowsEdit->setText( QString( "%1" ).arg( rows ) );
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::numberOfRows()
{
	return _rowsEdit->text().toInt();
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::distanceInverted()
{
    return _invertDistance->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setNumberOfColumns( int columns )
{
	_columnsEdit->setText( QString( "%1" ).arg( columns ) );
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::numberOfColumns()
{
	return _columnsEdit->text().toInt();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setNumberOfContours( int contours )
{
	_contoursEdit->setText( QString( "%1" ).arg( contours ) );
	_mapWidget->canvas()->setNumberOfContours( contours );
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::numberOfContours()
{
	return _contoursEdit->text().toInt();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setRiskRadius( double radius )
{
	int tmp = (int) radius;
	_riskSlider->blockSignals( true );
	_riskSlider->setValue( tmp );
	_riskSlider->blockSignals( false );
	_riskLabel->setText( QString( "%1 mm" ).arg( tmp ) );
	_mapWidget->canvas()->setRiskRadius( radius );
}

///////////////////////////////////////////////////////////////////////////
double QDistanceWidget::riskRadius()
{
	return (double) _riskSlider->value();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setRiskRadiusEnabled( bool enabled )
{
	_riskEnabled->blockSignals( true );
	_riskEnabled->setChecked( enabled );
	_riskEnabled->blockSignals( false );
	_mapWidget->canvas()->setRiskRadiusEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::riskRadiusEnabled()
{
	return _riskEnabled->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setUncertaintyEnabled( bool enabled )
{
    _uncertaintyEnabled->blockSignals( true );
    _uncertaintyEnabled->setChecked( enabled );
    _uncertaintyEnabled->blockSignals( false );
    //_mapWidget->canvas()->setUncertaintyEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::uncertaintyEnabled()
{
    return _uncertaintyEnabled->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setContoursEnabled( bool enabled )
{
	_contoursEnabled->blockSignals( true );
	_contoursEnabled->setChecked( enabled );
	_contoursEnabled->blockSignals( false );
	_mapWidget->canvas()->setContoursEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::contoursEnabled()
{
	return _contoursEnabled->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setProjectionEnabled( bool enabled )
{
	_projectionEnabled->blockSignals( true );
	_projectionEnabled->setChecked( enabled );
	_projectionEnabled->blockSignals( false );
	_mapWidget->canvas()->setProjectionEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::projectionEnabled()
{
	return _projectionEnabled->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setAutomaticViewPointsEnabled( bool enabled )
{
	_automaticViewPointsEnabled->blockSignals( true );
	_automaticViewPointsEnabled->setChecked( enabled );
	_automaticViewPointsEnabled->blockSignals( false );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::automaticViewPointsEnabled()
{
	return _automaticViewPointsEnabled->isChecked();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::setUpdateIsoValueEnabled( bool enabled )
{
	_updateIsoValue->blockSignals( true );
	_updateIsoValue->setChecked( enabled );
	_updateIsoValue->blockSignals( false );
}

///////////////////////////////////////////////////////////////////////////
bool QDistanceWidget::updateIsoValueEnabled()
{
	return _updateIsoValue->isChecked();
}

///////////////////////////////////////////////////////////////////////////
int QDistanceWidget::cameraRotationSpeed()
{
	return _speedSlider->value();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::compute()
{
	emit computeStarted();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::computeSingleDT()
{
    emit computeSingleDTStarted();
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::riskChanged( int value )
{
	double risk = (double) _riskSlider->value();
	_riskLabel->setText( tr( "%1 mm" ).arg( value ) );
	_mapWidget->canvas()->setRiskRadius( risk );
	emit riskRadiusChanged( risk );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::riskEnabled( bool enabled )
{
	double risk = (double) _riskSlider->value();
	_mapWidget->canvas()->setRiskRadius( risk );
	_mapWidget->canvas()->setRiskRadiusEnabled( enabled );
	emit riskRadiusEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::contoursEnabled( bool enabled )
{
	int nrContours = _contoursEdit->text().toInt();
	_mapWidget->canvas()->setContoursEnabled( enabled );
	_mapWidget->canvas()->setNumberOfContours( nrContours );
	emit distanceContoursEnabled( enabled, nrContours );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::uncertaintyEnabled( bool enabled )
{
    double interval = _probabilitySlider->value() / 100.0;
    _graphWidget->canvas()->setThresholdInterval(interval);
    emit distanceUncertaintyEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::colorLookupChanged(int index)
{
    emit colorLookupChanged(_colorLookupBox->currentText());
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::probabilityChanged(int value)
{
    QString text = QString("+/- %1 %").arg(value);
    _probabilityLabel->setText(text);
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::projectionEnabled( bool enabled )
{
	emit distanceProjectionEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::updateIsoValue( bool enabled )
{
	emit updateIsoValueEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::viewPointsEnabled( bool enabled )
{
	emit automaticViewPointsEnabled( enabled );
}

///////////////////////////////////////////////////////////////////////////
void QDistanceWidget::configChanged( const QString text )
{
	emit configurationChanged( text );
}

///////////////////////////////////////////////////////////////////////////
double QDistanceWidget::probabilityInterval()
{
    double value = _probabilitySlider->value() / 100.0;
    return value;
}

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceWidget * QDistanceWidget::graph()
{
	return _graphWidget;
}

///////////////////////////////////////////////////////////////////////////
QTumorMapWidget * QDistanceWidget::map()
{
	return _mapWidget;
}
