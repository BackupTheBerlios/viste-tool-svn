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
 * QVTKProgressCommand.cxx
 *
 * 2006-05-02	Tim Peeters
 * - First version
 *
 * 2006-05-15	Tim Peeters
 * - Added functions for getting/setting the parent widget.
 *
 * 2011-04-26	Evert van Aart
 * - Added optional overrides for the label and window title.
 *
 */


#ifndef bmia_QVTKProgressCommand_h
#define bmia_QVTKProgressCommand_h


/** Includes - VTK */

#include <vtkAlgorithm.h>
#include <vtkCommand.h>

/** Includes - Qt */

#include <QtGui/QApplication>
#include <QtGui/QProgressDialog>
#include <QDebug>
#include <QString>


namespace bmia {


/** A "vtkCommand" that can be used to monitor progress events. These events are 
	then displayed using a QProgressDialog. By default, the window title of this
	progress dialog is "DTITool"; this can be changed using "setTitle". The progress
	label will usually be fetched from the algorithm (e.g. the VTK filter); if the
	filter did not set a progress text, a default string will be used. The user can
	override this by calling "setLabel" with a custom (fixed) progress label. This 
	is especially useful for default VTK filters, which usually do not set any
	progress text; by calling "setLabel", the progress label can be changed to 
	something more descriptive than "Progress for algorithm [Algorithm Name]".
 */


class QVTKProgressCommand : public vtkCommand
{
	public:
		
		/** Constructor Call. */

		static QVTKProgressCommand * New();

		/** Updates the progress bar. An object of this class is added to an algorithm
			as an observer for progress events. Whenever this algorithm fires such
			a progress event (usually through "updateProgress()"), this "Execute"
			function is called. The "eventId" should always be the ID used for progress
			events. The "caller" is a pointer to the algorithm that fired the progress
			event, and the "callData" contains the new progress value (as a double in
			the range 0-1).
			@param caller		Algorithm that fired the progress event.
			@param eventId		Event identifier.
			@param callData		New progress value. */

		virtual void Execute(vtkObject * caller, unsigned long eventId, void * callData);

		/** Return a pointer to the progress dialog. */

		QProgressDialog * GetProgressDialog()
		{
			return this->ProgressDialog;
		}

		/** Set the parent widget for the progress dialog. If this function is 
			not called, the parent widget will be NULL.
			@param parent		Parent widget for the progress dialog. */
  
		void SetParentWidget(QWidget * parent);

		/** Return the parent widget of the progress dialog. */

		QWidget * GetParentWidget();

		/** Set a fixed label. If a non-empty string is passed, this string will
			always be used as the progress text (i.e., the progress text of the
			calling algorithm will be ignored. To clear this label override,
			call this function with an empty "QString".
			@param rLabel		Desired label. */

		void setLabel(QString rLabel);

		/** Set the title of the progress dialog. This will be "DTITool" by default. 
			@param rTitle		Desired title. */

		void setTitle(QString rTitle);

	protected:

		/** Constructor. */

		QVTKProgressCommand();

		/** Destructor. */

		~QVTKProgressCommand();

	private:
  
		/** Qt progress dialog used to show the progress. */

		QProgressDialog * ProgressDialog;

		/** Optional progress label. */

		QString label;

		/** Window title for the progress dialog. */

		QString title;

}; // class QVTKProgressCommand


} // namespace bmia


#endif // bmia_QVTKProgressCommand
