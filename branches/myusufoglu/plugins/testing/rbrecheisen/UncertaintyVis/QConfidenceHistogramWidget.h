#ifndef __QConfidenceHistogramWidget_h
#define __QConfidenceHistogramWidget_h

#include <QWidget>
#include <QPixmap>
#include <QPaintEvent>
#include <QMouseEvent>

class vtkConfidenceHistogram;
class vtkConfidenceInterval;

class QConfidenceHistogramWidget : public QWidget
{
	Q_OBJECT

public:

	QConfidenceHistogramWidget( QWidget * parent = 0 );
	virtual ~QConfidenceHistogramWidget();

	void setHistogram( vtkConfidenceHistogram * histogram );
	void setInterval( vtkConfidenceInterval * interval );
	void setOpacity( float opacity );

	void reset( bool update = true );

private:

	void paintEvent( QPaintEvent * event );
	void mouseMoveEvent( QMouseEvent * event );
	void mousePressEvent( QMouseEvent * event );
	void mouseReleaseEvent( QMouseEvent * event );

	QPointF toWidget( QPointF point );
	QPointF toNorm( QPointF point );

	int findAnchor( int x, int y );
	int findInterval( int x, int y );
	int findIntervalAnchor( int x, int y );

signals:

	void intervalChanged();
	void intervalSelected( int );

private:

	vtkConfidenceHistogram * histogram;
	vtkConfidenceInterval * interval;

	QPixmap * cache;
	QPointF selectedPoint;

	float opacity;
	int selectedInterval;
	bool modulatingValue;
	bool modulatingIntervalWidth;
};

#endif