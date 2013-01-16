// Includes DTI tool
#include <QMinimalDistanceCanvasWidget.h>
#include <QMinimalDistanceWidget.h>

// Includes C++
#include <iostream>

#define OFFSET_LEFT   30
#define OFFSET_RIGHT  15
#define OFFSET_TOP    10
#define OFFSET_BOTTOM 15

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceCanvasWidget::QMinimalDistanceCanvasWidget( QMinimalDistanceWidget * parent ) : QWidget()
{
	// Store reference to parent graph widget
	_parent = parent;
	_rangeX[0] = 0.0; _rangeX[1] = 1.0;
	_rangeY[0] = 0.0; _rangeY[1] = 1.0;
	_selectedDistance = 0.0;
    _thresholdInterval = 0.0;

	_levelPoint1.setX( 0.0f );
	_levelPoint1.setY( 0.0f );
	_levelPoint2.setX( 0.0f );
	_levelPoint2.setY( 0.0f );

	_area.setX( 0 );
	_area.setY( 0 );
	_area.setWidth( 0 );
	_area.setHeight( 0 );
}

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceCanvasWidget::~QMinimalDistanceCanvasWidget()
{
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::resizeEvent( QResizeEvent * event )
{
	_area.setX( OFFSET_LEFT );
	_area.setY( OFFSET_TOP );
	_area.setWidth( this->width() - OFFSET_LEFT - OFFSET_RIGHT );
	_area.setHeight( this->height() - OFFSET_TOP - OFFSET_BOTTOM );
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::paintEvent( QPaintEvent * event )
{
	QPainter p( this );

	// Draw background
	p.setPen( Qt::gray );
	p.fillRect( 0, 0, width(), height(), Qt::white );
	p.drawRect( 0, 0, width(), height() );

	if( _points.size() > 0 )
	{
		// Clear background
		p.fillRect( 0, 0, width(), height(), Qt::white );
		p.drawRect( 0, 0, width(), height() );

		// Draw grid lines
		int nrX = 5, nrY = 5;
		float stepX = (_rangeX[1] - _rangeX[0]) / 4.0f;
		float stepY = (_rangeY[1] - _rangeY[0]) / 4.0f;

		for( int i = 0; i < nrY; ++i )
		{
			QPointF p0( _rangeX[0], _rangeY[0] + i * stepY );
			QPointF p1( _rangeX[1], _rangeY[0] + i * stepY );
			QPoint  p2 = this->graphToWidget( p0 );
			QPoint  p3 = this->graphToWidget( p1 );

			QString label = QString( "%1" ).arg( _rangeY[0] + i * stepY, 0, 'f', 1 );
			QRect r = QFontMetrics( this->font() ).boundingRect( label );
			p.drawText( p2.x() - r.width() - 1, p2.y() + r.height() / 2 - 4, label );
			p.drawLine( p2, p3 );
		}

		for( int i = 0; i < nrX; ++i )
		{
			QPointF p0( _rangeX[0] + i * stepX, _rangeY[0] );
			QPointF p1( _rangeX[0] + i * stepX, _rangeY[1] );
			QPoint  p2 = this->graphToWidget( p0 );
			QPoint  p3 = this->graphToWidget( p1 );

			QString label = QString( "%1" ).arg( i * 0.25f, 0, 'f', 1 );
			QRect r = QFontMetrics( this->font() ).boundingRect( label );
			p.drawText( p2.x() - r.width() / 2, height() - 1, label );
			p.drawLine( p2, p3 );
		}

		// Set pen
		QPen pen( Qt::black );
		p.setPen( pen );
		p.setRenderHint( QPainter::Antialiasing );

		// Draw graph curve
		QPainterPath path;

		QPoint firstPoint;
		QPoint lastPoint;

		for( int i = 0; i < (_points.size() - 1); ++i )
		{
			QPoint p1 = this->graphToWidget( _points.at( i ) );
			if( i == 0 )
			{
				path.moveTo( p1 );
				firstPoint = p1;
			}
			else
			{
				QPoint p2 = this->graphToWidget( _points.at( i + 1 ) );
				path.lineTo( p2 );
				if( i + 1 == _points.size() - 1 )
				{
					lastPoint = p2;
				}
			}
		}

		// Draw graph
		pen.setColor( Qt::red );
		pen.setWidth( 3 );
		p.setPen( pen );
		p.drawPath( path );

		path.lineTo( QPoint( lastPoint.x(), firstPoint.y() ) );
		path.closeSubpath();

		QLinearGradient gradient( firstPoint.x(), lastPoint.y(), firstPoint.x(), firstPoint.y() );
		gradient.setColorAt( 0, QColor( 255, 0, 0, 255 ) );
		gradient.setColorAt( 1, QColor( 255, 0, 0, 0) );

		pen.setWidth( 1 );
		p.setPen( pen );
		p.fillPath( path, QBrush( gradient ) );

		// Draw uncertainty bounds
		pen.setColor( Qt::blue );
		pen.setWidth( 2 );
		p.setPen( pen );

		QPointF p0 = QPointF( _levelPoint1.x(), _rangeY[0] );
		QPointF p1 = QPointF( _levelPoint1.x(), _rangeY[1] );
		QPoint  p2 = this->graphToWidget( p0 );
		QPoint  p3 = this->graphToWidget( p1 );

		QPointF p4 = QPointF( _levelPoint2.x(), _rangeY[0] );
		QPointF p5 = QPointF( _levelPoint2.x(), _rangeY[1] );
		QPoint  p6 = this->graphToWidget( p4 );
		QPoint  p7 = this->graphToWidget( p5 );

		p.fillRect( QRect( p3, p6 ), QColor( 0, 0, 255, 64 ) );
		p.drawLine( p2, p3 );
		p.drawLine( p6, p7 );

		// Draw selected distance marker
		if( _selectedDistance > 0.0 )
		{
			float certainty = (_levelPoint1.x() - _rangeX[0]) / (_rangeX[1] - _rangeX[0]);

			QPointF p8 = QPointF( _levelPoint1.x(), _selectedDistance );
			QPoint  p9 = this->graphToWidget( p8 );
			QPainterPath circle;
			circle.addEllipse( p9, 4, 4 );

			QString label = QString( "%1-%2" ).arg( certainty, 0, 'f', 1 ).arg( p8.y(), 0, 'f', 1 );
			QRect r  = QFontMetrics( this->font() ).boundingRect( label );
			QRect r0 = QRect( 0, 0, r.width(), r.height() );
			QRect r1 = QRect( 0, 0, r.width() + 4, r.height() + 4 );

			int halfW = this->width() / 2;
			int halfH = this->height() / 2;

			if( p9.x() < halfW && p9.y() < halfH )
			{
				r1.moveTo( p9 );
			}
			else if( p9.x() < width() / 2 && p9.y() >= height() / 2 )
			{
				p9.setY( p9.y() - r1.height() );
				r1.moveTo( p9 );
			}
			else if( p9.x() >= width() / 2 && p9.y() < height() / 2 )
			{
				p9.setX( p9.x() - r1.width() );
				r1.moveTo( p9 );
			}
			else if( p9.x() >= width() / 2 && p9.y() >= height() / 2 )
			{
				p9.setX( p9.x() - r1.width() );
				p9.setY( p9.y() - r1.height() );
				r1.moveTo( p9 );
			}
			else {}

			r0.moveTo( QPoint( p9.x() + 2, p9.y() + 2 ) );

			pen.setColor( Qt::black );
			pen.setWidth( 1 );
			p.setPen( pen );

			// Draw label
			p.drawRect( r1 );
			p.fillRect( r1, QColor( 255, 255, 0, 128 ) );
			p.drawText( r0, label );

			// Draw dot at selected point
			p.fillPath( circle, Qt::black );
		}
	}

	p.end();
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::mouseReleaseEvent( QMouseEvent * event )
{
	int x = event->x();
	int y = event->y();
	QPointF p = this->widgetToGraph( QPoint( x, y ) );

	if( event->modifiers() == Qt::ShiftModifier )
	{
		_levelPoint2.setX( p.x() );
		_levelPoint2.setY( p.y() );
		emit rangeSelected( _levelPoint1, _levelPoint2 );
	}
	else
	{
		_levelPoint1.setX( p.x() );
		_levelPoint1.setY( p.y() );
		_levelPoint2.setX( p.x() );
		_levelPoint2.setY( p.y() );
		emit pointSelected( _levelPoint1 );
	}

	this->update();
}

///////////////////////////////////////////////////////////////////////////
QPoint QMinimalDistanceCanvasWidget::graphToArea( const QPointF & point )
{
	float x = point.x();
	float y = point.y();
	int ax = _area.width() * (x - _rangeX[0]) / (_rangeX[1] - _rangeX[0]);
	int ay = _area.height() * (_rangeY[1] - y) / (_rangeY[1] - _rangeY[0]);
	return QPoint( ax, ay );
}

///////////////////////////////////////////////////////////////////////////
QPointF QMinimalDistanceCanvasWidget::areaToGraph( const QPoint & point )
{
	float x = (float) point.x();
	float y = (float) point.y();
	float gx = _rangeX[0] + x * (_rangeX[1] - _rangeX[0]) / _area.width();
	float gy = _rangeY[0] + (_area.height() - y) * (_rangeY[1] - _rangeY[0]) / _area.height();
	return QPointF( gx, gy );
}

///////////////////////////////////////////////////////////////////////////
QPoint QMinimalDistanceCanvasWidget::graphToWidget( const QPointF & point )
{
	QPoint p = this->graphToArea( point );
	return this->areaToWidget( p );
}

///////////////////////////////////////////////////////////////////////////
QPoint QMinimalDistanceCanvasWidget::widgetToArea( const QPoint & point )
{
	int x = point.x();
	int y = point.y();
	int ax = x - OFFSET_LEFT;
	int ay = y - OFFSET_TOP;
	return QPoint( ax, ay );
}

///////////////////////////////////////////////////////////////////////////
QPoint QMinimalDistanceCanvasWidget::areaToWidget( const QPoint & point )
{
	int x = point.x();
	int y = point.y();
	int wx = x + OFFSET_LEFT;
	int wy = y + OFFSET_TOP;
	return QPoint( wx, wy );
}

///////////////////////////////////////////////////////////////////////////
QPointF QMinimalDistanceCanvasWidget::widgetToGraph( const QPoint & point )
{
	QPoint p = this->widgetToArea( point );
	return this->areaToGraph( p );
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setRangeX( double min, double max )
{
	_rangeX[0] = min;
	_rangeX[1] = max;

	_levelPoint1.setX( _rangeX[1] );
	_levelPoint2.setX( _rangeX[1] );
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setRangeY( double min, double max )
{
	_rangeY[0] = min;
	_rangeY[1] = max;
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setDistance( double distance )
{
	_selectedDistance = distance;
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setThreshold( double threshold )
{
	_levelPoint1.setX( threshold );
	_levelPoint1.setY( 0.0 );
	_levelPoint2.setX( threshold );
	_levelPoint2.setY( 0.0 );
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setThresholdInterval( double interval )
{
    _thresholdInterval;
}

///////////////////////////////////////////////////////////////////////////
void QMinimalDistanceCanvasWidget::setPoints( QList< QPointF > & points )
{
	_points.clear();
	for( int i = 0; i < points.size(); ++i )
		_points.append( points.at( i ) );
	this->update();
}

///////////////////////////////////////////////////////////////////////////
QList< QPointF > & QMinimalDistanceCanvasWidget::points()
{
	return _points;
}
