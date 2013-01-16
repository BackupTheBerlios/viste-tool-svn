#ifndef __QMinimalDistanceCanvasWidget_h
#define __QMinimalDistanceCanvasWidget_h

#include <QtGui>

class QMinimalDistanceWidget;

class QMinimalDistanceCanvasWidget : public QWidget
{
	Q_OBJECT

public:

	QMinimalDistanceCanvasWidget( QMinimalDistanceWidget * parent = 0 );
	virtual ~QMinimalDistanceCanvasWidget();

	void setPoints( QList< QPointF > & points );
	QList< QPointF > & points();

	void setRangeX( double min, double max );
	void setRangeY( double min, double max );
	void setThreshold( double threshold );
	void setDistance( double d );

    void setThresholdInterval(double interval);

private:

	void paintEvent( QPaintEvent * event );

	void mouseReleaseEvent( QMouseEvent * event );
	void resizeEvent( QResizeEvent * event );

	QPoint  graphToWidget( const QPointF & point );
	QPointF widgetToGraph( const QPoint & point );

	QPoint  graphToArea( const QPointF & point );
	QPointF areaToGraph( const QPoint & point );

	QPoint widgetToArea( const QPoint & point );
	QPoint areaToWidget( const QPoint & point );

signals:

	void pointSelected( QPointF & point );
	void rangeSelected( QPointF & point1, QPointF & point2 );

private:

	QMinimalDistanceWidget * _parent;

	double _rangeX[2];
	double _rangeY[2];

	double _selectedDistance;
    double _thresholdInterval;

	QList< QPointF > _points;

	QPointF _levelPoint1;
	QPointF _levelPoint2;
	QRect   _area;
};

#endif
