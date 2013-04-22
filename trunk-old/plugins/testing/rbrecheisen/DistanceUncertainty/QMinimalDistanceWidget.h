#ifndef __QMinimalDistanceWidget_h
#define __QMinimalDistanceWidget_h

#include <QMinimalDistanceCanvasWidget.h>
#include <QtGui>

class QMinimalDistanceWidget : public QWidget
{
	Q_OBJECT

public:

	QMinimalDistanceWidget( QWidget * parent = 0 );
	virtual ~QMinimalDistanceWidget();

	QSize sizeHint() const;
	QSizePolicy sizePolicy();

	QMinimalDistanceCanvasWidget * canvas();

private:

	QMinimalDistanceCanvasWidget * _canvas;
};

#endif
