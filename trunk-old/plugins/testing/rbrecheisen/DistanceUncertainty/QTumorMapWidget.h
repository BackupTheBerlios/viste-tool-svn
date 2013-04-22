#ifndef __QTumorMapWidget_h
#define __QTumorMapWidget_h

#include <QTumorMapCanvasWidget.h>
#include <QtGui>

class QTumorMapWidget : public QWidget
{
	Q_OBJECT

public:

	QTumorMapWidget( QWidget * parent = 0 );
	virtual ~QTumorMapWidget();

	QSize sizeHint() const;
	QSizePolicy sizePolicy();

	QTumorMapCanvasWidget * canvas();

private slots:

private:

	QTumorMapCanvasWidget * _canvas;
};

#endif
