#ifndef __QTumorMapCanvasWidget_h
#define __QTumorMapCanvasWidget_h

#include <QtGui>

class QTumorMapWidget;

class QTumorMapCanvasWidget : public QWidget
{
	Q_OBJECT

public:

	QTumorMapCanvasWidget( QTumorMapWidget * parent );
	virtual ~QTumorMapCanvasWidget();

    void setDistanceMap( double * mapA, double * mapB, int rows, int columns );
    void setDistanceMap( double * map, int rows, int columns );
	double * getDistanceMap();

    void setIndexMap( int * mapA, int * mapB, int rows, int columns );
	void setIndexMap( int * map, int rows, int columns );
	int * getIndexMap();

	void setMapRange( double min, double max );

    void setMinimalDistances(QList<double> & distances, QList<QPoint> & positions);

	void setMinimalDistance( double distance );
	void setRiskRadius( double radius );
	void setRiskRadiusEnabled( bool enabled );
	void setNumberOfContours( int numberOfContours );
	void setContoursEnabled( bool enabled );
	void setProjectionEnabled( bool enabled );

	void setSelectedVoxelIndex( int idx );
	void setSelectedDistancePoint( QPoint & p );

	QSize sizeHint() const;
	QSizePolicy sizePolicy();

private:

	void paintEvent( QPaintEvent * event );
	void mouseReleaseEvent( QMouseEvent * event );

	QPoint toWidget( const QPoint & p );
	QPoint toMap( const QPoint & p );

	QImage * createMapImage();
    QImage * createMapImageSpecial();

signals:

	void pointSelected( int );

private:

	QTumorMapWidget * _parent;
	QImage * _mapImage;
	QPoint _minDistPos;
	QPoint _selectedDistPos;

    QList<double> _distances;
    QList<QPoint> _positions;

    double * _map, * _mapA, * _mapB;
	double _riskRadius;
	double _mapRange[2];
	double _minDist;

    int * _indices, * _indicesA, * _indicesB;
    int _rows, _columns;
	int _numberOfContours;

	bool _riskRadiusEnabled;
	bool _contoursEnabled;
	bool _projectionEnabled;
};

#endif
