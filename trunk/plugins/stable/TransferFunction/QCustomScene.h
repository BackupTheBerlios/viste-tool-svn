/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
