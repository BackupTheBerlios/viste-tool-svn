/*
 * QLookUpTableWidget.h
 *
 * 2010-02-26	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-06	Evert van Aart
 * - Refactored code, added comments
 *
 */


#ifndef bmia_QLookUpTableWidget_h
#define bmia_QLookUpTableWidget_h


/** Includes - Qt */

#include <QtGui>


/** Forward Class Declarations */

class vtkColorTransferFunction;
class vtkPiecewiseFunction;


/** A small bar, shown below the main transfer function editor, which visualizes the
	current transfer function. If a piecewise function has been set, it is visualized using
	the opacity of the bar. 
*/

class QLookUpTableWidget : public QWidget
{
	Q_OBJECT

	public:

		/** Constructor */

		QLookUpTableWidget(QWidget * parent);

		/** Destructor */

		virtual ~QLookUpTableWidget();

		/** Sets the resolution of the table.
			@param fResolution	The number of sample points that will be used. */

		void setResolution(float fResolution);

		/** Sets the transfer function.
			@param pTf			Pointer to the transfer function. */

		void setTransferFunction(vtkColorTransferFunction * pTf);

		/** Sets the piecewise function.
			@param pPf			A pointer to the piecewise function. */

		void setPiecewiseFunction(vtkPiecewiseFunction * pPf);

		/** Sets the intensity range. Copied from the spin boxes in the GUI. 
			@param min			The minimum scalar value that will be in the range.
			@param max			The maximum scalar value that will be in the range. */

		void setIntensityRange(float min, float max);

	protected:

		/** Current transfer function. */

		vtkColorTransferFunction * m_pTf;

		/** Current piecewise function (if set). */

		vtkPiecewiseFunction * m_pPf;

		/** Range of intensity scalar values. */

		float m_fIntensityRange[2];

		/** Resolution of the bar, i.e., the number of points at which the output color
			of the transfer function is sampled. */

		float m_fResolution;

		/** Padding on the left and right of the bar. */

		int m_iPaddingLeft;
		int m_iPaddingRight;

		/** Horizontal and vertical range. */

		float m_fRangeX[2];
		float m_fRangeY[2];


		/** Handles the painting of the widget.
			@param e	Details of the paint event. */

		void paintEvent(QPaintEvent * e);

		/** Converts given point in normalized coordinates (0-1) to a position in widget coordinates.
			@param p	The point to be converted. */

		QPointF toAbs(QPointF & p);

};


#endif // bmia_QLookUpTableWidget_h
