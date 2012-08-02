/*
 * QLookUpTableWidget.cxx
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-06	Evert van Aart
 * - Refactored code, added comments
 *
 */


/** Includes */

#include "QLookUpTableWidget.h"
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>


//-----------------------------[ Constructor ]-----------------------------\\

QLookUpTableWidget::QLookUpTableWidget(QWidget * parent) : QWidget(parent),
																m_pTf(NULL),
																m_pPf(NULL),
																m_iPaddingLeft(0),
																m_iPaddingRight(0),
																m_fResolution(255.0f)
{
	// Initialize visible range
	m_fRangeX[0] = 0.0f;
	m_fRangeX[1] = 1.0f;
	m_fRangeY[0] = 0.0f;
	m_fRangeY[1] = 1.0f;

	// Initialize scalar range
	this->m_fIntensityRange[0] = 0.0f;
	this->m_fIntensityRange[1] = 1.0f;

	// Sets the maximum height of the widget
	setMaximumHeight(16);
}


//------------------------------[ Destructor ]-----------------------------\\

QLookUpTableWidget::~QLookUpTableWidget()
{

}


//----------------------------[ setResolution ]----------------------------\\

void QLookUpTableWidget::setResolution(float fResolution)
{
	this->m_fResolution = fResolution;
}


//--------------------------[ setIntensityRange ]--------------------------\\

void QLookUpTableWidget::setIntensityRange(float min, float max)
{
	// Store the variables and redraw the widget
	this->m_fIntensityRange[0] = min;
	this->m_fIntensityRange[1] = max;
	this->repaint();
}


//------------------------------[ paintEvent ]-----------------------------\\

void QLookUpTableWidget::paintEvent(QPaintEvent * e)
{
	// First call the event handler of the parent class
	QWidget::paintEvent(e);

	// Move the coordinate system, then flip it along the Y-axis
	QMatrix m;
	m.translate(0.0f, static_cast<float>(height()) - 1.0f);
	m.scale(1.0f, -1.0f);

	// Create the painter
	QPainter paint(this);

	// Set the painter options
	paint.setMatrix(m);
	paint.setMatrixEnabled(true);
	paint.setRenderHint(QPainter::Antialiasing, false);
	paint.setPen(Qt::NoPen);
	paint.setBrush(Qt::red);
	paint.drawRect(0, 0, 1, 1);

	float inc = 0.1f;

	// Create the checkered background
	for(int i = 0; i < 10; ++i)
	{
		if(i % 2)
		{
			QPointF p1(i * inc, 0.0f);
			QPointF p2((i + 1) * inc, 0.5f);
			paint.fillRect(QRectF(toAbs(p1), toAbs(p2)), Qt::lightGray);
		}
		else
		{
			QPointF p1( i * inc, 0.0f );
			QPointF p2( (i + 1) * inc, 0.5f );
			paint.fillRect(QRectF(toAbs(p1), toAbs(p2)), Qt::white);
		}

		if(i % 2)
		{
			QPointF p1(i * inc, 0.5f);
			QPointF p2((i + 1) * inc, 1.0f);
			paint.fillRect(QRectF(toAbs(p1), toAbs(p2)), Qt::white);
		}
		else
		{
			QPointF p1(i * inc, 0.5f);
			QPointF p2((i + 1) * inc, 1.0f);
			paint.fillRect(QRectF(toAbs(p1), toAbs(p2)), Qt::lightGray);
		}
	}

	// Draw the transfer function, if it exists
	if( m_pTf )
	{
		QPointF p1(0.0f, 0.0f);
		QPointF p2(1.0f, 0.0f);

		// Create a gradient over the width of the bar
		QLinearGradient grad(toAbs(p1), toAbs(p2));

		// Set the horizontal resolution of the bar
		int size = (int) this->m_fResolution;

		// Compute the range
		float range = this->m_fIntensityRange[1] - this->m_fIntensityRange[0];

		// Draw the bar form left to right
		for(int i = 0; i < size; i++)
		{
			// Fraction between zero and one
			float f = i / static_cast<float>(size - 1);

			// Compute the scalar value of the current horizontal position
			double intensity = this->m_fIntensityRange[0] + f * range;

			// If a piecewise function exists, use it to get the opacity
			float opacity = (this->m_pPf ? this->m_pPf->GetValue(intensity) : 1);

			// Map the intensity scalar value to a color using the transfer function
			QColor color;

			color.setRgbF(	this->m_pTf->GetRedValue(intensity),
							this->m_pTf->GetGreenValue(intensity),
							this->m_pTf->GetBlueValue(intensity),
							opacity									);

			// Add the color to the gradient
			grad.setColorAt( f, color );
		}

		QPointF p3(0.0f, 0.0f);
		QPointF p4(1.0f, 1.0f);

		// Fill the bar using the computed gradient
		paint.fillRect(QRectF(toAbs(p3), toAbs(p4)), QBrush(grad));
	}

	paint.setMatrixEnabled(false);
}


//--------------------------------[ toAbs ]--------------------------------\\

QPointF QLookUpTableWidget::toAbs(QPointF & p)
{
	// Convert point with coordinates between zero and one to real coordinates
	float fx = (p.x() - m_fRangeX[0]) / (m_fRangeX[1] - m_fRangeX[0]) * (static_cast<float>(width()) - m_iPaddingLeft - m_iPaddingRight) + m_iPaddingLeft;
	float fy = (p.y() - m_fRangeY[0]) / (m_fRangeY[1] - m_fRangeY[0]) *  static_cast<float>(height());
	return QPointF(fx, fy);
}


//-------------------------[ setTransferFunction ]-------------------------\\

void QLookUpTableWidget::setTransferFunction(vtkColorTransferFunction * pTf)
{
	// Store the transfer function pointer, repaint the bar
	this->m_pTf = pTf;
	this->repaint();
}


//-------------------------[ setPiecewiseFunction ]------------------------\\

void QLookUpTableWidget::setPiecewiseFunction(vtkPiecewiseFunction * pPf)
{
	// Store the piecewise function pointer, repaint the bar
	this->m_pPf = pPf;
	this->repaint();
}

