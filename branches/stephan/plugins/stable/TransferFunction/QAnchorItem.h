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
