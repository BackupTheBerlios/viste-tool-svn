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
 * TransferFunctionCanvas.h
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-06	Evert van Aart
 * - Refactored code, added comments
 *
 */


#ifndef bmia_QTransferFunctionCanvas_h
#define bmia_QTransferFunctionCanvas_h


/** Includes - Qt */

#include <QGraphicsView>
#include <QDebug>

/** Includes - VTK */

#include <vtkImageData.h>


/** Forward Class Declarations */

class vtkColorTransferFunction;
class vtkPiecewiseFunction;
class QCustomScene;


/** Canvas containing the transfer function editor. Actual drawing is done 
	in the scene object of type "QCustomScene".
*/

class QTransferFunctionCanvas : public QGraphicsView
{
	Q_OBJECT

	public:

		/** Constructor */

		QTransferFunctionCanvas(QWidget * parent = 0);

		/** Set the intensity range. 
			@param min			Minimum intensity.
			@param max			Maximum intensity. */

		void setIntensityRange(double min, double max);

		/** Copy the data set pointer to the scene object. 
			@param pData		Image data set.
			@param useSecondMax	Use second-largest value for normalization. */

		void setDataSet(vtkImageData * pData, bool useSecondMax);

		/** Copy the transfer function pointer to the scene object. 
			@param tf			Transfer function pointer. */

		void setTransferFunction(vtkColorTransferFunction * tf);

		/** Copy the piecewise function pointer to the scene object. 
			@param pf			Piecewise function pointer. */

		void setPiecewiseFunction(vtkPiecewiseFunction * pf);

	protected:

		/** Scene object. */

		QCustomScene * pScene;

		/** Called when the canvas is resized. 
			@param event		Event details, not used. */

		void resizeEvent(QResizeEvent * event);

	signals:

		/** Emitted by "slotTransferFunctionChanged". */

		void transferFunctionChanged();

	private slots:

		/** Triggered by the scene object when the transfer function changes. */

		void slotTransferFunctionChanged();

};


#endif // bmia_QTransferFunctionCanvas_h
