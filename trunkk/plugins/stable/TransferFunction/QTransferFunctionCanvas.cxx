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
 * TransferFunctionCanvas.cxx
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-06	Evert van Aart
 * - Refactored code, added comments
 *
 */


/** Includes */

#include "QTransferFunctionCanvas.h"
#include "QCustomScene.h"


//-----------------------------[ Constructor ]-----------------------------\\

QTransferFunctionCanvas::QTransferFunctionCanvas(QWidget* parent) : QGraphicsView(parent)
{
	// Create a new scene
	pScene = new QCustomScene(this);
	this->setScene(pScene);
	this->setDragMode(QGraphicsView::RubberBandDrag);

	// Connect the "transferFunctionChanged" signal of the scene object to the function in this class
	connect(this->pScene, SIGNAL(tranferFunctionChanged()), this, SLOT(slotTransferFunctionChanged()));

	// Create a palette
	QPalette palette;
	palette.setBrush(QPalette::Base, QBrush(QColor(255, 255, 221)));
	this->setPalette(palette);
}


//-----------------------------[ resizeEvent ]-----------------------------\\

void QTransferFunctionCanvas::resizeEvent(QResizeEvent * event)
{
	// Do nothing with the event details
	Q_UNUSED(event);

	// Resize the canvas
	this->fitInView(this->sceneRect());
}


//--------------------------[ setIntensityRange ]--------------------------\\

void QTransferFunctionCanvas::setIntensityRange(double min, double max)
{
	// Create a new rectangle based on the minimum and maximum values
	QRectF rect;
	rect.setLeft(min * 1000);
	rect.setRight(max * 1000);
	rect.setTop(0.0);
	rect.setBottom(1000.0);

	// Store the rectangle as the scene rectangle, and resize the canvas
	this->pScene->setSceneRect(rect);
	this->fitInView(this->sceneRect());
	this->pScene->updateLine();
}


//------------------------------[ setDataSet ]-----------------------------\\

void QTransferFunctionCanvas::setDataSet(vtkImageData * pData, bool useSecondMax)
{
	// Copy the settings to the scene object
	this->pScene->setDataSet(pData, useSecondMax);
}


//-------------------------[ setTransferFunction ]-------------------------\\

void QTransferFunctionCanvas::setTransferFunction(vtkColorTransferFunction * tf)
{
	// Copy the settings to the scene object
	this->pScene->setTransferFunction(tf);
}


//-------------------------[ setPiecewiseFunction ]------------------------\\

void QTransferFunctionCanvas::setPiecewiseFunction(vtkPiecewiseFunction * pf)
{
	// Copy the settings to the scene object
	this->pScene->setPiecewiseFunction(pf);
}


//---------------------[ slotTransferFunctionChanged ]---------------------\\

void QTransferFunctionCanvas::slotTransferFunctionChanged()
{
	emit this->transferFunctionChanged();
}

