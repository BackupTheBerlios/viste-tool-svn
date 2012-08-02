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
