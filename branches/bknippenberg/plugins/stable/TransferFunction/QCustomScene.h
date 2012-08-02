/*
 * QCustomScene.h
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-04	Evert van Aart
 * - Added comments
 *
 * 2011-03-28	Evert van Aart
 * - Prevented divide-by-zero errors for scalar images with zero range. 
 *
 */


#ifndef QCUSTOMSCENE_H
#define QCUSTOMSCENE_H


/** Includes - Qt */

#include <QGraphicsScene>
#include <QMenu>
#include <QTimer>
#include <QPainter>
#include <QColor>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>

/** Includes - Custom Files */

#include "QAnchorItem.h"


class QCustomScene : public QGraphicsScene
{
    Q_OBJECT

	public:
    
		/** Pointer to the selected anchor item. */

		QAnchorItem * itemClicked;

		/** Current transfer function. */
    
		vtkColorTransferFunction * pTf;

		/** Current piecewise function. */

		vtkPiecewiseFunction * pPf;

		/** Constructor */
    
		QCustomScene(QObject * parent = 0);

		/** Rebuild the line between the anchor points. */

		void updateLine();

		/** Recreate the transfer function, based on the anchor point locations. */

		void updateTransferFunction();

		/** Change the current scalar volume data set, and update the histogram. 
			@param pData		New image data.
			@param useSecondMax	Normalize using second-largest value instead of largest value. */

		void setDataSet(vtkImageData * pData, bool useSecondMax);

		/** Used for sorting the anchor items based on their X position. 
			@param a			First anchor item.
			@param b			Second anchor item. */

		static bool anchorlessThan(QAnchorItem * a, QAnchorItem * b)
		{
			return a->scenePos().x() < b->scenePos().x();
		}

		/** Set a new transfer function pointer. Creates anchor points.
			@param tf			New transfer function. */

		void setTransferFunction(vtkColorTransferFunction * tf);

		/** Set a new piecewise function, the size of which should match that of the
			current transfer function pointer. Updates the Y position of the 
			existing anchor points. */

		void setPiecewiseFunction(vtkPiecewiseFunction * pf);

	protected:

		/** Used to automatically update the transfer function after some time. */

		QTimer timer;

		/** List of anchor points. */

		QList<QAnchorItem *> anchors;

		/** Line connecting the anchor points. */

		QPainterPath * pLine;

		/** Histogram for the current image. */

		QPainterPath * pHistoGram;

		/** Context menu shown when right-clicking an anchor point. */
    
		QMenu anchorMenu;

		/** Draw the widget.
			@param painter		Painter object used to draw the widget.
			@param rect			Target area. */

		void drawBackground(QPainter * painter, const QRectF & rect);

		/** Called when the mouse is clicked anywhere within the widget area.
			@param event		Event details. */

		void mousePressEvent(QGraphicsSceneMouseEvent * event);

		/** Called when the mouse is dragged when over the widget area. 
			@param mouseEvent	Event details. */

		void mouseMoveEvent(QGraphicsSceneMouseEvent * mouseEvent);

	signals:
    
		/** Called when the timer times out. */

		void tranferFunctionChanged();

	private slots:
    
		/** Called when the timer times out. */
		
		void timeout();

};

#endif // QCustomScene_H
