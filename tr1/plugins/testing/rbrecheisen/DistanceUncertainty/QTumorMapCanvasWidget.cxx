#include <QTumorMapCanvasWidget.h>
#include <QTumorMapWidget.h>

#include <iostream>

///////////////////////////////////////////////////////////////////////////
QTumorMapCanvasWidget::QTumorMapCanvasWidget( QTumorMapWidget * parent ) : QWidget( parent )
{
	_parent = parent;
	_map = 0;
	_mapRange[0] = 0.0;
	_mapRange[1] = 0.0;
	_mapImage = 0;
	_indices = 0;
	_riskRadius = 0.0;
	_riskRadiusEnabled = false;
	_contoursEnabled = false;
	_projectionEnabled = false;
	_minDistPos.setX( -1 );
	_minDistPos.setY( -1 );
	_minDist = 0;
	_selectedDistPos.setX( -1 );
	_selectedDistPos.setY( -1 );
	_numberOfContours = 8;
	_rows = 0;
	_columns = 0;
}

///////////////////////////////////////////////////////////////////////////
QTumorMapCanvasWidget::~QTumorMapCanvasWidget()
{
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::paintEvent( QPaintEvent * event )
{
	QPainter p( this );
	p.setRenderHint( QPainter::Antialiasing );

	if( _mapImage )
	{
		//QImage image = _mapImage->scaled( this->size() );
		p.drawImage( 0, 0, (* _mapImage) );
	}

	if( _minDistPos.x() >= 0 && _minDistPos.y() >= 0 )
	{
		int x = _minDistPos.x();
		int y = _minDistPos.y();

		QPen pen;
		pen.setWidth( 2 );
		pen.setColor( Qt::yellow );
		p.setPen( pen );
		p.drawLine( x - 5, y, x + 5, y );
		p.drawLine( x, y - 5, x, y + 5 );

		QString label = QString( "%1" ).arg( _minDist, 0, 'f', 1 );
		QRect r  = QFontMetrics( this->font() ).boundingRect( label );
		QRect rr = QRect( x + 10, y, r.width(), r.height() );
		p.drawText( rr, label );
	}

	if( _selectedDistPos.x() >= 0 && _selectedDistPos.y() >= 0 )
	{
		int x = _selectedDistPos.x();
		int y = _selectedDistPos.y();

		QPainterPath path;
		path.addEllipse( x - 4, y - 4, 8, 8 );

		p.setPen( Qt::NoPen );
		p.fillPath( path, QBrush( Qt::yellow ) );

		if( _map )
		{
			QPen pen;
			pen.setWidth( 2 );
			pen.setColor( Qt::yellow );
			p.setPen( pen );

			double distance = _map[y * _columns + x];
			QString label = QString( "%1" ).arg( distance, 0, 'f', 1 );
			QRect r  = QFontMetrics( this->font() ).boundingRect( label );
			QRect rr = QRect( x + 10, y, r.width(), r.height() );
			p.drawText( rr, label );
		}
	}

    if(_distances.count() > 0 && _positions.count() > 0)
    {
        for(int i = 0; i < _positions.count(); ++i)
        {
            QPoint point = _positions.at(i);
            QPen pen;
            pen.setWidth(2);
            pen.setColor(Qt::yellow);
            p.setPen(pen);
            p.drawLine(point.x() - 5, point.y(), point.x() + 5, point.y());
            p.drawLine(point.x(), point.y() - 5, point.x(), point.y() + 5);
        }
    }

	p.end();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::mouseReleaseEvent( QMouseEvent * event )
{
	if( _indices == 0 )
		return;

	int x = event->x();
	int y = event->y();
	QPoint p = this->toMap( QPoint( x, y ) );

	_selectedDistPos.setX( p.x() );
	_selectedDistPos.setY( p.y() );

	emit pointSelected( _indices[p.y() * _columns + p.x()] );
	this->update();
}

//#include <iostream>

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setSelectedVoxelIndex( int idx )
{
	if( _indices == 0 )
		return;

	for( int k = 0; k < _rows * _columns; ++k )
	{
		if( _indices[k] == idx )
		{
			int j = (int) floor( (double) idx / _columns );
			int i = (int) floor( (double) idx - j * _columns );
			_selectedDistPos.setX( j );
			_selectedDistPos.setY( i );
			break;
		}
	}

	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setSelectedDistancePoint( QPoint & p )
{
	_selectedDistPos.setX( p.x() );
	_selectedDistPos.setY( p.y() );
	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setDistanceMap( double * mapA, double * mapB, int rows, int columns )
{
    if( _mapA )
        delete [] _mapA;
    _mapA = new double[rows*columns];
    memcpy( _mapA, mapA, rows*columns*sizeof(double));

    if( _mapB )
        delete [] _mapB;
    _mapB = new double[rows*columns];
    memcpy( _mapB, mapB, rows*columns*sizeof(double));

    _rows = rows;
    _columns = columns;
    _selectedDistPos.setX(-1);
    _selectedDistPos.setY(-1);

    if( _mapImage )
        delete _mapImage;
    _mapImage = this->createMapImageSpecial();
    this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setDistanceMap( double * map, int rows, int columns )
{
	if( _map )
		delete [] _map;
	_map = new double[rows * columns];
	memcpy( _map, map, rows * columns * sizeof( double ) );

	_rows = rows;
	_columns = columns;
	_selectedDistPos.setX( -1 );
	_selectedDistPos.setY( -1 );

	if( _mapImage )
		delete _mapImage;
	_mapImage = this->createMapImage();

	this->update();
}

///////////////////////////////////////////////////////////////////////////
double * QTumorMapCanvasWidget::getDistanceMap()
{
	return _map;
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setIndexMap( int * mapA, int * mapB, int rows, int columns )
{
    if( _indicesA )
        delete [] _indicesA;
    _indicesA = new int[rows * columns];
    memcpy( _indicesA, mapA, rows * columns * sizeof( int ) );

    if( _indicesB )
        delete [] _indicesB;
    _indicesB = new int[rows * columns];
    memcpy( _indicesB, mapB, rows * columns * sizeof( int ) );

    _rows = rows;
    _columns = columns;
    _selectedDistPos.setX( -1 );
    _selectedDistPos.setY( -1 );

    this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setIndexMap( int * map, int rows, int columns )
{
	if( _indices )
		delete [] _indices;
	_indices = new int[rows * columns];
	memcpy( _indices, map, rows * columns * sizeof( int ) );

	_rows = rows;
	_columns = columns;
	_selectedDistPos.setX( -1 );
	_selectedDistPos.setY( -1 );

	this->update();
}

///////////////////////////////////////////////////////////////////////////
int * QTumorMapCanvasWidget::getIndexMap()
{
	return _indices;
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setMinimalDistances(QList<double> & distances, QList<QPoint> & positions)
{
    _distances.clear();
    _positions.clear();

    for(int i = 0; i < distances.count(); ++i)
        _distances.append(distances.at(i));

    for(int i = 0; i < positions.count(); ++i)
        _positions.append(positions.at(i));

    this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setMapRange( double min, double max )
{
	_mapRange[0] = min;
	_mapRange[1] = max;
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setMinimalDistance( double distance )
{
	_minDist = distance;
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setRiskRadius( double radius )
{
	_riskRadius = radius;

	if( _mapImage )
		delete _mapImage;
	_mapImage = this->createMapImage();

	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setContoursEnabled( bool enabled )
{
	_contoursEnabled = enabled;

	if( _mapImage )
		delete _mapImage;
	_mapImage = this->createMapImage();

	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setNumberOfContours( int numberOfContours )
{
	if( _numberOfContours != numberOfContours )
	{
		_numberOfContours = numberOfContours;

		if( _mapImage )
			delete _mapImage;
		_mapImage = this->createMapImage();

		this->update();
	}
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setRiskRadiusEnabled( bool enabled )
{
	_riskRadiusEnabled = enabled;

	if( _mapImage )
		delete _mapImage;
	_mapImage = this->createMapImage();

	this->update();
}

///////////////////////////////////////////////////////////////////////////
void QTumorMapCanvasWidget::setProjectionEnabled( bool enabled )
{
	_projectionEnabled = enabled;
	this->update();
}

///////////////////////////////////////////////////////////////////////////
QPoint QTumorMapCanvasWidget::toWidget( const QPoint & point )
{
	return QPoint( point.x(), point.y() );
}

///////////////////////////////////////////////////////////////////////////
QPoint QTumorMapCanvasWidget::toMap( const QPoint & point )
{
	return QPoint( point.x(), point.y() );
}

///////////////////////////////////////////////////////////////////////////
QImage * QTumorMapCanvasWidget::createMapImageSpecial()
{
    if( ! _mapA || ! _mapB )
        return 0;

    QImage * image = new QImage( _columns, _rows, QImage::Format_RGB32 );
    double step = 1.0 / _numberOfContours;
    double min = 99999999.0;

    for( int i = 0; i < _rows; ++i )
    {
        for( int j = 0; j < _columns; ++j )
        {
            QColor color;
            int idx = i * _columns + j;
            double value = (_mapA[idx] - _mapRange[0]) / (_mapRange[1] - _mapRange[0]);

            if( value < 0.0 )
                value = 0.0;
            if( value > 1.0 )
                value = 1.0;

            if( _mapA[idx] < _riskRadius && _riskRadiusEnabled )
            {
                color.setRgbF( 1.0, 0.0, 0.0 );
            }
            else if( _mapB[idx] < _riskRadius && _riskRadiusEnabled )//&& _riskRadiusUncertaintyEnabled )
            {
                color.setRgbF( 0.0, 0.0, 1.0 );
            }
            else
            {
                if( _contoursEnabled )
                {
                    int level = (int) floor( value / step );
                    double newValue = level * step;
                    color.setRgbF( newValue, newValue, newValue );
                }
                else
                {
                    color.setRgbF( value, value, value );
                }
            }

            if( _map[idx] < min )
            {
                min = _map[idx];
                _minDistPos.setX( j );
                _minDistPos.setY( i );
            }

            image->setPixel( j, i, color.rgb() );
        }
    }

    return image;
}

///////////////////////////////////////////////////////////////////////////
QImage * QTumorMapCanvasWidget::createMapImage()
{
	if( ! _map )
		return 0;

	QImage * image = new QImage( _columns, _rows, QImage::Format_RGB32 );
	double step = 1.0 / _numberOfContours;
	double min = 99999999.0;

	for( int i = 0; i < _rows; ++i )
	{
		for( int j = 0; j < _columns; ++j )
		{
			QColor color;
			int idx = i * _columns + j;

			double value = (_map[idx] - _mapRange[0]) / (_mapRange[1] - _mapRange[0]);
			if( value < 0.0 )
				value = 0.0;
			if( value > 1.0 )
				value = 1.0;

			if( _map[idx] < _riskRadius && _riskRadiusEnabled )
			{
				color.setRgbF( 1.0, 0.0, 0.0 );
			}
			else
			{
				if( _contoursEnabled )
				{
					int level = (int) floor( value / step );
					double newValue = level * step;
					color.setRgbF( newValue, newValue, newValue );
				}
				else
				{
					color.setRgbF( value, value, value );
				}
			}

			if( _map[idx] < min )
			{
				min = _map[idx];
				_minDistPos.setX( j );
				_minDistPos.setY( i );
			}

			image->setPixel( j, i, color.rgb() );
		}
	}

	return image;
}

///////////////////////////////////////////////////////////////////////////
QSize QTumorMapCanvasWidget::sizeHint() const
{
	return _parent->sizeHint();
};

///////////////////////////////////////////////////////////////////////////
QSizePolicy QTumorMapCanvasWidget::sizePolicy()
{
	return _parent->sizePolicy();
}
