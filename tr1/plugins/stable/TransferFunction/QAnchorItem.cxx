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
 * QAnchorIem.cxx
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-04	Evert van Aart
 * - Refactored code, added comments
 *
 */


/** Includes */

#include "QAnchorItem.h"
#include "QCustomScene.h"


//-----------------------------[ Constructor ]-----------------------------\\

QAnchorItem::QAnchorItem(QGraphicsItem * parent) : QGraphicsEllipseItem(parent)
{
	// Set option flags
	this->setFlags(	QGraphicsItem::ItemIsMovable |
					QGraphicsItem::ItemIgnoresTransformations |
					QGraphicsItem::ItemIsSelectable					);

#if QT_VERSION >= 0x040600
	  setFlag(QGraphicsItem::ItemSendsGeometryChanges);
	  setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
#endif

	// Set graphical options
	this->setCacheMode(QGraphicsItem::NoCache);
	this->setCursor(Qt::OpenHandCursor);
	this->setZValue(2);
	this->setRect(QRectF(-6,-6,12,12));
	this->setBrush(QBrush(Qt::green));
}


//---------------------------------[ type ]--------------------------------\\

int QAnchorItem::type() const
{
    return Type;
}


//------------------------------[ itemChange ]-----------------------------\\

QVariant QAnchorItem::itemChange(GraphicsItemChange change, const QVariant & value)
{
	// Position of the anchor item has changed
	if (change == ItemPositionChange && scene())
	{
		// Get new position of the item
		QPointF newPos = value.toPointF();

		// Get the background rectangle
		QRectF rect = scene()->sceneRect();

		// Use a default position if the new position is not located within the background rectangle
		if (!rect.contains(newPos))
		{
			newPos.setX(qMin(rect.right(), qMax(newPos.x(), rect.left())));
			newPos.setY(qMin(rect.bottom(), qMax(newPos.y(), rect.top())));

			if (((QCustomScene *) this->scene())->pPf == NULL)
			{
				newPos.setY(500);
			}

			return newPos;
		}

		// If no pointwise function has been set, all anchors have the same Y position
		if (((QCustomScene*) this->scene())->pPf == NULL)
		{
			newPos.setY(500);
		}

		return newPos;
	}

	// Change in selection
	else if (change == ItemSelectedChange)
	{
		// Change pen depending on whether or not the anchor is selected
		if (value.toBool())
		{
			this->setPen(QPen(Qt::black, 3));
		}
		else
		{
			this->setPen(QPen(Qt::black, 1));
		}
	}

	// Use the parent function to handle the rest of the item changes
	return QGraphicsItem::itemChange(change, value);
}


//---------------------------[ openColorPicker ]---------------------------\\

void QAnchorItem::openColorPicker()
{
	// Get a new color using the color dialog
    QColor color = QColorDialog::getColor(this->brush().color(), 0);

	// Change the brush
    if (color.isValid())
    {
        this->setBrush(color);
    }
}


//---------------------------------[ paint ]-------------------------------\\

void QAnchorItem::paint(QPainter * painter, const QStyleOptionGraphicsItem * option, QWidget * widget)
{
	// Enable anti-aliasing
    painter->setRenderHint(QPainter::Antialiasing);

	// Paint the item using the parent function
    QGraphicsEllipseItem::paint(painter,option,widget);
}


//---------------------------[ mousePressEvent ]---------------------------\\

void QAnchorItem::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
	// If the right mouse button was pressed, select the item
    if (event->button() == Qt::RightButton)
    {
        ((QCustomScene *) this->scene())->itemClicked = this;
    }
	// For the left mouse button, change the cursor to the dragging cursor
    else if (event->button() == Qt::LeftButton)
    {
        this->setCursor(Qt::ClosedHandCursor);
    }

	// Use the parent handler to handle the event
    QGraphicsItem::mousePressEvent(event);
}


//--------------------------[ mouseReleaseEvent ]--------------------------\\

void QAnchorItem::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
	// Change the cursor
    this->setCursor(Qt::OpenHandCursor);

	// Use the parent handler to handle the event
	QGraphicsItem::mouseReleaseEvent(event);
}


//----------------------------[ mouseMoveEvent ]---------------------------\\

void QAnchorItem::mouseMoveEvent(QGraphicsSceneMouseEvent * event)
{
	// Show coordinates in tooltip
    if (((QCustomScene *) this->scene())->pPf == NULL)
	{
        QToolTip::showText(QCursor::pos(), QString::number(this->scenePos().x() / 1000.0, 'f' ,2));
	}
    else
	{
        QToolTip::showText(QCursor::pos(), QString::number(this->scenePos().x() / 1000.0, 'f' ,2) + "/" + QString::number((1000 - this->scenePos().y()) / 1000.0, 'f' , 2));
	}

	// Use the parent handler to handle the event
    QGraphicsItem::mouseMoveEvent(event);

	// Update the transfer function
	((QCustomScene*) this->scene())->updateTransferFunction();
}
