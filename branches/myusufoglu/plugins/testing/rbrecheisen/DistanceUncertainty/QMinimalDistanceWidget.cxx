// Includes DTI tool
#include <QMinimalDistanceWidget.h>

// Includes C++
#include <iostream>

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceWidget::QMinimalDistanceWidget( QWidget * parent ) : QWidget( parent )
{
	_canvas = new QMinimalDistanceCanvasWidget( this );

	QVBoxLayout * layout = new QVBoxLayout;
	layout->setContentsMargins( 1, 1, 1, 1 );
	layout->addWidget( _canvas );

	this->setLayout( layout );
}

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceWidget::~QMinimalDistanceWidget()
{
	delete _canvas;
}

///////////////////////////////////////////////////////////////////////////
QSize QMinimalDistanceWidget::sizeHint() const
{
	return QSize( 600, 200 );
};

///////////////////////////////////////////////////////////////////////////
QSizePolicy QMinimalDistanceWidget::sizePolicy()
{
	return QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Fixed );
}

///////////////////////////////////////////////////////////////////////////
QMinimalDistanceCanvasWidget * QMinimalDistanceWidget::canvas()
{
	return _canvas;
}
