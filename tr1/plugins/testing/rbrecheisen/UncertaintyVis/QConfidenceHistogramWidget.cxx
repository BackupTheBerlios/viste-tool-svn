#include <QConfidenceHistogramWidget.h>

#include <QPainter>
#include <QPen>

#include <vtkConfidenceHistogram.h>
#include <vtkConfidenceInterval.h>
#include <vtkConfidenceIntervalProperties.h>

#define BORDER_LEFT		20.0f
#define BORDER_RIGHT	20.0f
#define BORDER_BOTTOM	20.0f
#define BORDER_TOP		20.0f
#define ANCHOR_SIZE		8.0f
#define BIN_OFFSET		2.0f

///////////////////////////////////////////////////////////////////////////
QConfidenceHistogramWidget::QConfidenceHistogramWidget( QWidget * parent ) :
	QWidget( parent )
{
	this->histogram = 0;
	this->interval = 0;
	this->cache = 0;
	this->opacity = 0.5f;
	this->selectedInterval = -1;
	this->modulatingValue = false;
	this->modulatingIntervalWidth = false;

	this->selectedPoint.setX( -1 );
	this->selectedPoint.setY( -1 );
	this->setMinimumSize( 200, 200 );
}

///////////////////////////////////////////////////////////////////////////
QConfidenceHistogramWidget::~QConfidenceHistogramWidget()
{
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::setHistogram( vtkConfidenceHistogram * histogram )
{
	this->histogram = histogram;
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::setInterval( vtkConfidenceInterval * interval )
{
	this->interval = interval;
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::setOpacity( float opacity )
{
	this->opacity = opacity;
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::reset( bool update )
{
	if( this->cache )
		delete this->cache;
	this->cache = 0;
	if( update )
		this->update();
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::paintEvent( QPaintEvent * event )
{
	QPen pen;
	pen.setWidth( 2 );
	pen.setStyle( Qt::SolidLine );
	pen.setBrush( Qt::black );

	// Draw static (cached) graphics

	if( this->cache == 0 || this->cache->rect() != this->rect() )
	{
		if( this->cache )
			delete this->cache;
		this->cache = new QPixmap( this->rect().size() );
		this->cache->fill( Qt::white );

		QPainter p( this->cache );
		QPointF p1 = this->toWidget( QPointF( 0.0f, 0.0f ) );
		QPointF p2 = this->toWidget( QPointF( 1.0f, 0.0f ) );
		QPointF p3 = this->toWidget( QPointF( 0.0f, 1.0f ) );
		p.setPen( pen );
		p.drawLine( p1, p2 );
		p.drawLine( p1, p3 );

		if( this->histogram )
		{
			vtkConfidenceHistogram * h = this->histogram;
			int nrBins = h->GetNumberOfBins();
			float * minMax = h->GetMinMax();
			float * values = h->GetProbabilities();

			if( values )
			{
				float binWidth = 1.0f / nrBins;
				for( int i = 0; i < nrBins; ++i )
				{
					QPointF p1 = this->toWidget( QPointF(
							i / static_cast< float >( nrBins ),
									std::min( values[i], minMax[1] ) / minMax[1] ) );
					p1.setX( p1.x() + BIN_OFFSET );

					QPointF p2 = this->toWidget( QPointF(
							i / static_cast< float >( nrBins ) + binWidth, 0.0f ) );
					p2.setX( p2.x() - BIN_OFFSET );

					QRectF r( p1, p2 );
					p.fillRect( r, Qt::black );
					p.drawRect( r );
				}
			}
		}
	}

	// Draw dynamic graphics

	QPainter p( this );
	p.setPen( pen );
	p.drawPixmap( 0, 0, * this->cache );

	if( this->interval )
	{
		vtkConfidenceInterval * ci = this->interval;

		if( ci->GetNumberOfIntervals() > 0 )
		{
			float * totalRange = ci->GetRange();
			float delta = totalRange[1] - totalRange[0];

			for( int i = 0; i < ci->GetNumberOfIntervals(); ++i )
			{
				float * range = ci->GetInterval( i );
				float * color = ci->GetProperties()->GetColor( i );
				float value   = ci->GetProperties()->GetValue( i );

				float tmp = this->opacity;
				if( ci->GetProperties()->IsEnabled( i ) == false )
				{
					tmp = 0.0f;
				}

				int intColor[4];
				intColor[0] = static_cast< int >( 255 * color[0] );
				intColor[1] = static_cast< int >( 255 * color[1] );
				intColor[2] = static_cast< int >( 255 * color[2] );
				intColor[3] = static_cast< int >( 255 * tmp );

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QPointF p1 = this->toWidget( QPointF( range[0], value ) );
				QPointF p2 = this->toWidget( QPointF( range[1], 0.0f ) );
				QRectF r( p1, p2 );

				p.fillRect( r, QColor(
						intColor[0],
						intColor[1],
						intColor[2], intColor[3] ) );
				p.drawRect( r );

				QVector< QPointF > points;
				p1 = QPointF( (range[1] + range[0]) / 2.0f, value );
				p2 = this->toWidget( p1 );
				points.append( QPointF( p2.x() + ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() - ANCHOR_SIZE ) );
				points.append( QPointF( p2.x() - ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() + ANCHOR_SIZE ) );
				QPolygonF polygon( points );
				p.drawPolygon( polygon );
			}

			if( this->selectedInterval > -1 )
			{
				float * range = ci->GetInterval( this->selectedInterval );
				float value = ci->GetProperties()->GetValue( this->selectedInterval );

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QPointF p1 = this->toWidget( QPointF( range[0], value ) );
				QPointF p2 = this->toWidget( QPointF( range[1], 0.0f ) );
				QRectF r( p1, p2 );

				pen.setWidth( 3 );
				pen.setBrush( Qt::yellow );
				p.setPen( pen );
				p.drawRect( r );
			}

			for( int i = 0; i < ci->GetNumberOfIntervals(); ++i )
			{
				float * range = ci->GetInterval( i );
				float value = ci->GetProperties()->GetValue( i );

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QVector< QPointF > points;
				QPointF p1( (range[1] + range[0]) / 2.0f, value );
				QPointF p2 = this->toWidget( p1 );
				points.append( QPointF( p2.x() + ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() - ANCHOR_SIZE ) );
				points.append( QPointF( p2.x() - ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() + ANCHOR_SIZE ) );
				QPolygonF polygon( points );

				pen.setWidth( 2 );
				pen.setBrush( Qt::black );
				p.setPen( pen );
				p.setBrush( Qt::yellow );
				p.drawPolygon( polygon );
				p.setBrush( Qt::NoBrush );
			}

			for( int i = 0; i < ci->GetNumberOfIntervals() - 1; ++i )
			{
				float * range = ci->GetInterval( i );
				float tmp1 = ci->GetProperties()->GetValue( i );
				float tmp2 = ci->GetProperties()->GetValue( i + 1 );
				float value = (tmp1 < tmp2 ) ? tmp1 : tmp2;

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QVector< QPointF > points;
				QPointF p1( range[1], value / 2.0f );
				QPointF p2 = this->toWidget( p1 );
				points.append( QPointF( p2.x() + ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() - ANCHOR_SIZE ) );
				points.append( QPointF( p2.x() - ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() + ANCHOR_SIZE ) );
				QPolygonF polygon( points );

				pen.setWidth( 2 );
				pen.setBrush( Qt::black );
				p.setPen( pen );
				p.setBrush( Qt::yellow );
				p.drawPolygon( polygon );
				p.setBrush( Qt::NoBrush );
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::mousePressEvent( QMouseEvent * event )
{
	this->selectedInterval = -1;
	this->selectedPoint.setX( -1 );
	this->selectedPoint.setY( -1 );
	emit intervalSelected( this->selectedInterval );

	this->modulatingValue = false;
	this->modulatingIntervalWidth = false;

	int index = this->findAnchor( event->x(), event->y() );
	if( index > -1 )
	{
		this->selectedInterval = index;
		this->selectedPoint.setX( event->x() );
		this->selectedPoint.setY( event->y() );
		this->modulatingValue = true;
		emit intervalSelected( this->selectedInterval );
	}
	else
	{
		index = this->findIntervalAnchor( event->x(), event->y() );
		if( index > -1 )
		{
			this->selectedInterval = index;
			this->selectedPoint.setX( event->x() );
			this->selectedPoint.setY( event->y() );
			this->modulatingIntervalWidth = true;	
			emit intervalSelected( this->selectedInterval );
		}
		else
		{
			index = this->findInterval( event->x(), event->y() );
			if( index > -1 )
			{
				this->selectedInterval = index;
				this->selectedPoint.setX( event->x() );
				this->selectedPoint.setY( event->y() );
				emit intervalSelected( this->selectedInterval );
			}
		}
	}

	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::mouseReleaseEvent( QMouseEvent * event )
{
	if( this->modulatingValue || this->modulatingIntervalWidth )
	{
		if( this->interval )
			this->interval->SetChanged( true );
		emit intervalChanged();
	}

	this->selectedPoint.setX( -1 );
	this->selectedPoint.setY( -1 );
	this->modulatingValue = false;
	this->modulatingIntervalWidth = false;
}

///////////////////////////////////////////////////////////////////////////
void QConfidenceHistogramWidget::mouseMoveEvent( QMouseEvent * event )
{
	if( this->modulatingValue )
	{
		QPointF p1 = this->toNorm( QPointF( event->x(), event->y() ) );
		QPointF p2 = this->toNorm( this->selectedPoint );
		float dy = p1.y() - p2.y();

		vtkConfidenceInterval * ci = this->interval;
		float value = ci->GetProperties()->GetValue( this->selectedInterval );
		value += dy;
		if( value > 1.0f ) value = 1.0f;
		if( value < 0.0f ) value = 0.0f;
		ci->GetProperties()->SetValue( this->selectedInterval, value );

		this->selectedPoint.setX( event->x() );
		this->selectedPoint.setY( event->y() );
		this->update();
	}
	else if( this->modulatingIntervalWidth )
	{
		QPointF p1 = this->toNorm( QPointF( event->x(), event->y() ) );
		QPointF p2 = this->toNorm( this->selectedPoint );
		float dx = p1.x() - p2.x();

		vtkConfidenceInterval * ci = this->interval;
		float * totalRange = ci->GetRange();
		float * range1 = ci->GetInterval( this->selectedInterval );
		float * range2 = ci->GetInterval( this->selectedInterval + 1 );

		dx *= (totalRange[1] - totalRange[0]);
		range1[1] += dx;
		range2[0] += dx;

		if( range1[1] > range2[1] ) range1[1] = range2[1];
		if( range1[1] < range1[0] ) range1[1] = range1[0];
		if( range2[0] < range1[0] ) range2[0] = range1[0];
		if( range2[0] > range2[1] ) range2[0] = range2[1];

		ci->SetInterval( this->selectedInterval, range1[0], range1[1] );
		ci->SetInterval( this->selectedInterval + 1, range2[0], range2[1] );
		this->selectedPoint.setX( event->x() );
		this->selectedPoint.setY( event->y() );
		this->update();
	}
	else
	{
	}
}

///////////////////////////////////////////////////////////////////////////
QPointF QConfidenceHistogramWidget::toWidget( QPointF point )
{
	float graphWidth  = this->width() - BORDER_LEFT - BORDER_RIGHT;
	float graphHeight = this->height() - BORDER_TOP - BORDER_BOTTOM;
	float x = point.x() * graphWidth + BORDER_LEFT;
	float y = (1.0f - point.y()) * graphHeight + BORDER_TOP;
	return QPointF( x, y );
}

///////////////////////////////////////////////////////////////////////////
QPointF QConfidenceHistogramWidget::toNorm( QPointF point )
{
	float graphWidth  = this->width() - BORDER_LEFT - BORDER_RIGHT;
	float graphHeight = this->height() - BORDER_TOP - BORDER_BOTTOM;
	float x =   (point.x() - BORDER_LEFT) / graphWidth;
	float y = -((point.y() - BORDER_TOP) / graphHeight) - 1.0f;
	return QPointF( x, y );
}

///////////////////////////////////////////////////////////////////////////
int QConfidenceHistogramWidget::findAnchor( int x, int y )
{
	if( this->interval )
	{
		if( this->interval->GetNumberOfIntervals() > 0 )
		{
			float * totalRange = this->interval->GetRange();
			float delta = totalRange[1] - totalRange[0];

			for( int i = 0; i < this->interval->GetNumberOfIntervals(); ++i )
			{
				float * range = this->interval->GetInterval( i );
				float value = this->interval->GetProperties()->GetValue( i );

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QVector< QPointF > points;
				QPointF p1( (range[1] + range[0]) / 2.0f, value );
				QPointF p2 = this->toWidget( p1 );
				points.append( QPointF( p2.x() + ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() - ANCHOR_SIZE ) );
				points.append( QPointF( p2.x() - ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() + ANCHOR_SIZE ) );

				QPolygonF polygon( points );
				QPointF p( x, y );
				if( polygon.containsPoint( p, Qt::OddEvenFill ) )
				{
					return i;
				}
			}
		}
	}

	return -1;
}

///////////////////////////////////////////////////////////////////////////
int QConfidenceHistogramWidget::findInterval( int x, int y )
{
	if( this->interval )
	{
		if( this->interval->GetNumberOfIntervals() > 0 )
		{
			float * totalRange = this->interval->GetRange();
			float delta = totalRange[1] - totalRange[0];

			for( int i = 0; i < this->interval->GetNumberOfIntervals(); ++i )
			{
				float * range = this->interval->GetInterval( i );
				float value = this->interval->GetProperties()->GetValue( i );

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QPointF p1 = this->toWidget( QPointF( range[0], value ) );
				QPointF p2 = this->toWidget( QPointF( range[1], 0.0f ) );
				QRectF r( p1, p2 );
				if( r.contains( x, y ) )
				{
					return i;
				}
			}
		}
	}

	return -1;
}

///////////////////////////////////////////////////////////////////////////
int QConfidenceHistogramWidget::findIntervalAnchor( int x, int y )
{
	if( this->interval )
	{
		if( this->interval->GetNumberOfIntervals() > 0 )
		{
			float * totalRange = this->interval->GetRange();
			float delta = totalRange[1] - totalRange[0];

			for( int i = 0; i < this->interval->GetNumberOfIntervals() - 1; ++i )
			{
				float * range = this->interval->GetInterval( i );
				float tmp1  = this->interval->GetProperties()->GetValue( i );
				float tmp2  = this->interval->GetProperties()->GetValue( i + 1 );
				float value = (tmp1 < tmp2) ? tmp1 : tmp2;

				range[0] = (range[0] - totalRange[0]) / delta;
				range[1] = (range[1] - totalRange[0]) / delta;
				value    = (value - totalRange[0]) / delta;

				QVector< QPointF > points;
				QPointF p1( range[1], value / 2.0f );
				QPointF p2 = this->toWidget( p1 );
				points.append( QPointF( p2.x() + ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() - ANCHOR_SIZE ) );
				points.append( QPointF( p2.x() - ANCHOR_SIZE, p2.y() ) );
				points.append( QPointF( p2.x(), p2.y() + ANCHOR_SIZE ) );

				QPolygonF polygon( points );
				QPointF p( x, y );
				if( polygon.containsPoint( p, Qt::OddEvenFill ) )
				{
					return i;
				}
			}
		}
	}

	return -1;
}
