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
