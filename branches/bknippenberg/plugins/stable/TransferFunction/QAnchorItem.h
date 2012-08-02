/*
 * QAnchorIem.h
 *
 * 2010-04-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-04	Evert van Aart
 * - Refactored code, added comments
 *
 */


#ifndef QANCHORITEM_H
#define QANCHORITEM_H


/** Includes - Qt */

#include <QGraphicsEllipseItem>
#include <QBrush>
#include <QCursor>
#include <QColorDialog>
#include <QGraphicsSceneMouseEvent>
#include <QToolTip>
#include <QDebug>
#include <QPainter>


/** Custom Qt class defining the behavior of the graphical anchor elements. */

class QAnchorItem : public QGraphicsEllipseItem
{
	public:
    
		/** Constructor */

		QAnchorItem(QGraphicsItem * parent = 0);

		/** Enumeration for the object type. */

		enum 
		{ 
			Type = UserType + 1 
		};

		/** Return the object type. */

		int type() const;

		/** Open a color picking dialog for the current anchor. */

		void openColorPicker();


	protected:
    
		/** Called when the state of the item changes.
			@param change	Parameter of the item that is changing. 
			@param value	New value of the parameter. */

		QVariant itemChange(GraphicsItemChange change, const QVariant & value);

		/** Draws anchor object.
			@param painter	Painter to be used for drawing. 
			@param option	Style options for the item.
			@param widget	Widget that is being painted on. */

		void paint(QPainter * painter, const QStyleOptionGraphicsItem * option, QWidget * widget = 0);

		/** Called when the mouse is pressed while on the anchor object.
			@param event	Event details. */

		void mousePressEvent(QGraphicsSceneMouseEvent * event);

		/** Called when the mouse is released.
		@param event	Event details. */

		void mouseReleaseEvent(QGraphicsSceneMouseEvent * event);

		/** Called when the mouse is moved while the left button is pressed down. 
		@param event	Event details. */
   
		void mouseMoveEvent(QGraphicsSceneMouseEvent * event);

}; // class QAnchorItem


#endif // QANCHORITEM_H
