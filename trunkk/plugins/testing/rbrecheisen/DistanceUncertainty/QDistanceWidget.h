#ifndef __QDistanceWidget_h
#define __QDistanceWidget_h

#include <QtGui>

#include <QTumorMapWidget.h>
#include <QMinimalDistanceWidget.h>

class QDistanceWidget : public QTabWidget
{
	Q_OBJECT

public:

	QDistanceWidget( QWidget * parent = 0 );
	virtual ~QDistanceWidget();

	void addDataset( const QString name );
	void addPolyDataset( const QString name );
	void addConfiguration( QString name );

	int selectedFiberIndex();
	int selectedTumorIndex();
    int selectedVolumeIndex();
    int selectedTumorMeshIndex();

	QString selectedFiberName();
	QString selectedTumorName();
	QString selectedTumorMeshName();
	QString selectedColorLookupTableName();
    QString selectedVolumeName();

	void setNumberOfSamples( int number );
	int numberOfSamples();

	void setNumberOfRows( int rows );
	int numberOfRows();

	void setNumberOfColumns( int columns );
	int numberOfColumns();

	void setNumberOfContours( int number );
	int numberOfContours();

	void setRiskRadius( double radius );
	double riskRadius();

	void setRiskRadiusEnabled( bool enabled );
	bool riskRadiusEnabled();

    void setContoursEnabled( bool enabled );
	bool contoursEnabled();

	void setProjectionEnabled( bool enabled );
	bool projectionEnabled();

	void setAutomaticViewPointsEnabled( bool enabled );
	bool automaticViewPointsEnabled();

	void setUpdateIsoValueEnabled( bool enabled );
	bool updateIsoValueEnabled();

    void setUncertaintyEnabled( bool enabled );
    bool uncertaintyEnabled();

    bool distanceInverted();

	int cameraRotationSpeed();

    double probabilityInterval();

	QTumorMapWidget * map();
	QMinimalDistanceWidget * graph();

private slots:

	void compute();	
	void riskEnabled( bool );
	void riskChanged( int );
	void contoursEnabled( bool );
    void uncertaintyEnabled( bool );
	void projectionEnabled( bool );
	void configChanged( const QString );
    void probabilityChanged(int);
	void viewPointsEnabled( bool );
	void updateIsoValue( bool );
    void colorLookupChanged(int);

    void computeSingleDT();

signals:

	void computeStarted();
	void configurationChanged( QString );
	void riskRadiusChanged( double );
	void riskRadiusEnabled( bool );
	void distanceContoursEnabled( bool, int );
	void distanceProjectionEnabled( bool );
    void distanceUncertaintyEnabled( bool );
	void automaticViewPointsEnabled( bool );
	void updateIsoValueEnabled( bool );
    void colorLookupChanged(QString);

    void computeSingleDTStarted();

private:

	QPushButton * _button;
    QPushButton * _buttonComputeDT;

	QComboBox * _fiberBox;
	QComboBox * _tumorBox;
    QComboBox * _volumeBox;
    QComboBox * _tumorMeshBox;
	QComboBox * _configurationBox;
	QComboBox * _colorLookupBox;

	QLabel * _riskLabel;
    QLabel * _probabilityLabel;

	QSlider * _riskSlider;
	QSlider * _speedSlider;
    QSlider * _probabilitySlider;

	QCheckBox * _riskEnabled;
	QCheckBox * _contoursEnabled;
    QCheckBox * _uncertaintyEnabled;
	QCheckBox * _projectionEnabled;
	QCheckBox * _automaticViewPointsEnabled;
	QCheckBox * _updateIsoValue;
    QCheckBox * _invertDistance;

	QLineEdit * _rowsEdit;
	QLineEdit * _columnsEdit;
	QLineEdit * _samplesEdit;
	QLineEdit * _contoursEdit;

	QTumorMapWidget * _mapWidget;
	QMinimalDistanceWidget * _graphWidget;
};

#endif
