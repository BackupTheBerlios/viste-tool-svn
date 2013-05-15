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
 * QCustomScene.cxx
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


/** Includes */

#include "QCustomScene.h"
#include "QAnchorItem.h"



//-----------------------------[ Constructor ]-----------------------------\\

QCustomScene::QCustomScene(QObject * parent) : QGraphicsScene(parent)
{
	// Initialize background rectangle
    this->setSceneRect(0, 0, 255, 1000);

	// Actions displayed when right-clicking on an anchor
    QAction * actionColor  = new QAction("Change Color",  this);
    QAction * actionDelete = new QAction("Delete Anchor", this);

	// Create anchor context menu
    this->anchorMenu.addAction(actionColor);
    this->anchorMenu.addAction(actionDelete);

	// Initialize pointers
    this->pLine = new QPainterPath;
    this->pHistoGram = NULL;
    this->setDataSet(NULL, false);
    this->itemClicked = NULL;
    this->pTf = NULL;
    this->pPf = NULL;

	// Initialize timer
    this->timer.setInterval(40);
    this->timer.setSingleShot(true);

	// Connect the timeout signal of the timer to the appropriate function
    connect(&(this->timer), SIGNAL(timeout()), this, SLOT(timeout()));
}


//----------------------------[ drawBackground ]---------------------------\\

void QCustomScene::drawBackground(QPainter * painter, const QRectF & rect)
{
	Q_UNUSED(rect);

	// Enable anti-aliasing, disable the pen
	painter->setRenderHint(QPainter::Antialiasing);
	painter->setPen(Qt::NoPen);

	// Draw the histogram if possible
	if (this->pHistoGram != NULL)
	{
		painter->setBrush(QColor(Qt::red).lighter(130));
		painter->drawPath(*this->pHistoGram);
	}

	// Draw the grid with anti-aliasing turned off
	painter->setPen(Qt::gray);
	painter->setRenderHint(QPainter::Antialiasing, false);

	for( int i = 1; i < 5; ++i)
	{
		double p =  i / 5.0;
		painter->drawLine( QPointF(this->sceneRect().left() , p * this->height()) , QPointF(this->sceneRect().right(), p * this->height()));
		painter->drawLine( QPointF(this->sceneRect().left() + p * (this->width()), 0.0), QPointF(this->sceneRect().left() + p * (this->width()), this->height()));
	}

	// Turn anti-aliasing back on
	painter->setRenderHint(QPainter::Antialiasing,true);
	
	// Create a new pen
	QPen pen;
	pen.setColor(Qt::black);
	pen.setCosmetic(true);
	pen.setWidth(3);
	painter->setPen(pen);
	painter->setBrush(Qt::NoBrush);

	// Draw the transfer function line using the new pen
	painter->drawPath(*this->pLine);
}


//---------------------------[ mousePressEvent ]---------------------------\\

void QCustomScene::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
	// Do nothing if no transfer function has been selected
    if (this->pTf == NULL)
		return;

	// First, use the parent event handler
	QGraphicsScene::mousePressEvent(event);

	if (event->button() == Qt::RightButton)
	{
		// Display the right-click context menu
		if (itemClicked)
		{
			QAction * action =  this->anchorMenu.exec(QCursor::pos());
			
			// Open a color picker dialog to select a new color for the anchor point
			if (action != NULL && action->text() == "Change Color")
			{
				itemClicked->openColorPicker();
			}
			// Delete an existing anchor point
			else if (action != 0 && action->text() == "Delete Anchor" && this->anchors.length() > 2)
			{
				this->removeItem(itemClicked);
				this->anchors.removeOne(itemClicked);
				this->updateLine();
			}
		}
		// Create a new anchor point
		else
		{
			QAnchorItem * pAnchor = new QAnchorItem();

			// Vertical position is locked when no piecewise function has been set
			if (this->pPf == NULL)
			{
				pAnchor->setPos(event->scenePos().x(),500);
			}
			else
			{
				pAnchor->setPos(event->scenePos());
			}

			// Add the new anchor to the lists and re-draw the line
			this->addItem(pAnchor);
			this->anchors.append(pAnchor);
			this->updateLine();
		}

	} // if [right-clicked]

	this->updateTransferFunction();
	this->itemClicked = NULL;
}


//------------------------------[ updateLine ]-----------------------------\\

void QCustomScene::updateLine()
{
	// Delete any existing line
	if (!pLine->isEmpty())
	{
		delete pLine;
		pLine = new QPainterPath;
	}

	// Do nothing if no anchors have been added
	if (this->anchors.isEmpty())
		return;

	// Sort the anchors based on their positions
	qStableSort(this->anchors.begin(), this->anchors.end(), QCustomScene::anchorlessThan);

	// Move to some position far to the left of the scene
	pLine->moveTo(this->sceneRect().left() - 1000,this->anchors.first()->scenePos().y());

	// Loop through all anchor points
	foreach(QAnchorItem * anchor, this->anchors)
	{
		// Connect the anchor point to the existing line
		pLine->lineTo(anchor->scenePos());
	}

	// Move to a position far to the right of the scene
	pLine->lineTo(this->sceneRect().right() + 10000, this->anchors.last()->scenePos().y());

	// Update the scene
	update();
}


//------------------------[ updateTransferFunction ]-----------------------\\

void QCustomScene::updateTransferFunction()
{
	// Do nothing if no transfer function has been set
	if (this->pTf == NULL)
		return;

	// Remove all existing points of the transfer function
	this->pTf->RemoveAllPoints();

	// Loop through all anchor points
	foreach(QAnchorItem * anchor, this->anchors)
	{
		// Add the X position and the color of the anchor point to the transfer function
		this->pTf->AddRGBPoint(	anchor->scenePos().x() / 1000.0,
								anchor->brush().color().redF(),
								anchor->brush().color().greenF(),
								anchor->brush().color().blueF());

	}

	// If no piecewise function exists, we're done here
	if (this->pPf == NULL)
	{
		emit this->tranferFunctionChanged();
		return;
	}

	// Otherwise, remove all existing points from the piecewise function
	this->pPf->RemoveAllPoints();

	// Loop through all anchor points
	foreach(QAnchorItem * anchor, this->anchors)
	{
		// Add the coordinates of the anchor point
		this->pPf->AddPoint(anchor->scenePos().x() / 1000.0, (1000.0 - anchor->scenePos().y()) / 1000.0);
	}

	// Start the timer
	if (!this->timer.isActive())
	{
		this->timer.start();
	}
}


//----------------------------[ mouseMoveEvent ]---------------------------\\

void QCustomScene::mouseMoveEvent(QGraphicsSceneMouseEvent * mouseEvent)
{
	// Use the parent event handler, and then re-build the line
    QGraphicsScene::mouseMoveEvent(mouseEvent);
    updateLine();
}


//------------------------------[ setDataSet ]-----------------------------\\

void QCustomScene::setDataSet(vtkImageData * pData, bool useSecondMax)
{
	// Do nothing if no image data has been selected
    if (pData == NULL)
        return;

	// Histogram bins
	double bins[256];

	// Delete existing histogram, and create a new one
    if (this->pHistoGram != NULL)
        delete this->pHistoGram;

    this->pHistoGram = new QPainterPath;

	// Set all bins to zero
    for(int i = 0; i < 256; ++i)
	{
        bins[i] = 0;
	}

    // Get the number of points in the image
    int numberOfPoints = pData->GetNumberOfPoints();

    // Get the scalar range of the input image
    double range[2];
    pData->GetScalarRange(range);

	// If the range is zero, we can't create a histogram
	if ((range[1] - range[0]) == 0.0)
	{
		return;
	}

	// Get the scalar array
	vtkDataArray * scalars = pData->GetPointData()->GetScalars();

	// Check if the scalar array exists
	if (!scalars)
		return;

	// Current point value
    double tempValue;

	// Scaled scalar value, between zero and one
    double d01;

	// Loop through all points, and fill up the histogram
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		tempValue = scalars->GetTuple1(ptId);
		d01 = (tempValue - range[0]) / (range[1] - range[0]);
		bins[(int) (d01 * 255)]++;
	}

	// Largest and second largest histogram value
    double secondMax = -9999;
    double max       = -9999;

	// Loop through all histogram bins
    for(int i = 0; i < 256; ++i)
    {
		// Divide the bin value by the number of points in the image
        bins[i] = bins[i] / numberOfPoints;

		// Check if the bin value is larger than the current second-largest value
        if (bins[i] > secondMax)
        {
			// If it is larger than the second-largest value, but smaller than
			// the largest value, update the second-largest value.

            if (bins[i] < max)
			{
                secondMax = bins[i];
			}

			// Otherwise, make the second-largest value equal to the current largest value,
			// and overwrite the largest value with the current bin value. 

            else
            {
                secondMax = max;
                max = bins[i];
            }
        }

    } // for [all bins]

	// Use either the largest or second-largest value for normalization
    double tempMax = (useSecondMax) ? (secondMax) : (max);

	// Move cursor to the start
    this->pHistoGram->moveTo(1000 * range[0], 1000);

	// Loop through all histogram bins
    for(int i = 0; i < 256; ++i)
    {
		// Add a line for the current bin
        double x = 1000 * (range[0] + ( i / 256.0) * (range[1] - range[0]));
        double y = 1000 - bins[i] / tempMax * 1000;
        this->pHistoGram->lineTo(x, y);
    }

	// Finalize creation of the histogram
    this->pHistoGram->lineTo(1000 * range[1], 1000);
    this->pHistoGram->closeSubpath();
    this->update();
}


//-------------------------[ setTransferFunction ]-------------------------\\

void QCustomScene::setTransferFunction(vtkColorTransferFunction * tf)
{
	// Store the transfer function pointer
    this->pTf = tf;

	// Remove all existing anchors
    foreach(QAnchorItem * anchor,this->anchors)
    {
        this->removeItem(anchor);
    }

    this->anchors.clear();

	// Do nothing if no transfer function has been selected
    if (tf == NULL)
        return;

	// Values of current transfer function point
    double node[6];

	// Number of anchor points in the new transfer function
    int count = tf->GetSize();

	// Loop through all points
    for(int i = 0; i < count; ++i)
    {
		// Get the six values of the current point
        tf->GetNodeValue(i, node);

		// Create a new anchor point
        QAnchorItem * pAnchor = new QAnchorItem();

		// Position depends on first value of "node"
        pAnchor->setPos(node[0] * 1000, 500);

		// Color depends on second, third and fourth values of "node"
        QColor color;
        color.setRedF(node[1]);
        color.setGreenF(node[2]);
        color.setBlueF(node[3]);
        pAnchor->setBrush(color);

		// Add anchor to the list, re-draw the line
        this->addItem(pAnchor);
        this->anchors.append(pAnchor);
        this->updateLine();
    }
}


//-------------------------[ setPiecewiseFunction ]------------------------\\

void QCustomScene::setPiecewiseFunction(vtkPiecewiseFunction * pf)
{
	// Store the pointer
    this->pPf = pf;

    if (pf != NULL)
    {
		// Piecewise function should always have the same size as the transfer function
        Q_ASSERT(this->pPf->GetSize() == this->pTf->GetSize());

		// Re-draw the line
        this->updateLine();

		// Y position of anchor
        double value;

		// Loop through all existing anchor points
        foreach(QAnchorItem * anchor, this->anchors)
        {
			// Use the piecewise function to compute the Y position of the anchor
            value = this->pPf->GetValue(anchor->scenePos().x() / 1000.0);
            anchor->setPos(anchor->scenePos().x(), 1000 - value * 1000);
        }
    }

	// Redraw the line
    this->updateLine();
}


//-------------------------------[ timeout ]-------------------------------\\

void QCustomScene::timeout()
{
	// On timer timeout, update the transfer function
    emit this->tranferFunctionChanged();
}
