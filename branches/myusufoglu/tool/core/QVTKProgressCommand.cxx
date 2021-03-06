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


/** Includes */

#include "QVTKProgressCommand.h"


namespace bmia {



//---------------------------[ Constructor Call ]--------------------------\\

QVTKProgressCommand * QVTKProgressCommand::New()
{
	return new QVTKProgressCommand;
}


//-----------------------------[ Constructor ]-----------------------------\\

QVTKProgressCommand::QVTKProgressCommand()
{
	// Create a new progress dialog
	this->ProgressDialog = new QProgressDialog("", NULL, 0, 100);
	this->ProgressDialog->setRange(0, 100);

	// Only show the dialog if the action takes more than 500 milliseconds
	this->ProgressDialog->setMinimumDuration(500);

	// The default title if "DTITool"
	this->title = "DTITool";

	// Label override is empty by default
	this->label = QString();
}


//------------------------------[ Destructor ]-----------------------------\\

QVTKProgressCommand::~QVTKProgressCommand()
{
	// Delete the progress dialog
	delete this->ProgressDialog;
}


//-------------------------------[ Execute ]-------------------------------\\

void QVTKProgressCommand::Execute(vtkObject * caller, unsigned long eventId, void * callData)
{
	// Get the progress, and scale it to the range 0-100.
	double progress = *(static_cast<double*>(callData)) * 100.0;

	// Get the algorithm pointer
	vtkAlgorithm * alg = vtkAlgorithm::SafeDownCast(caller);

	// Set the window title. This is "DTITool" by default, but can be changed
	// through "setTitle" (to, for example, the name of the plugin that created
	// the corresponding algorithm).

	this->ProgressDialog->setWindowTitle(this->title);

	// If the label has been set by the user, we always show this label
	if (this->label.isEmpty() == false)
	{
		this->ProgressDialog->setLabelText(this->label);
	}
	// Otherwise, we need to construct the label here
	else
	{
		QString progressText;
  
		// If the caller is an algorithm...
		if (alg)
		{
			// ...and it has a progress text set...
			if (alg->GetProgressText())
			{
				// ...the progress dialog label will be this algorithm progress text.
				progressText.append(alg->GetProgressText());
			}
			// If the algorithm does not have progress text set...
			else
			{
				// ...use the class name to construct a default progress label.
				progressText.append("Progress for algorithm '");
				progressText.append(caller->GetClassName());
				progressText.append("'");
			}
		} // if [Caller is an algorithm]
	
		// If the caller is not an algorithm...
		else
		{
			// ...use a different default progress label.
			progressText.append("Progress for VTK object ");
			progressText.append(caller->GetClassName());
		}

		alg = NULL;

		// Set the progress dialog label
		this->ProgressDialog->setLabelText(progressText);
	}
  
	// Set the actual progress
	this->ProgressDialog->setValue((int) progress);

	// Show the progress dialog
	if (progress < 100.0) 
		this->ProgressDialog->show();
  
	qApp->processEvents();
}


//---------------------------[ SetParentWidget ]---------------------------\\

void QVTKProgressCommand::SetParentWidget(QWidget * parent)
{
	// Tim: The parent widget of a Qt widget cannot be changed after construction
	// (or I could not find the function for that). Therefore, we delete the 
	// old progress dialog widget and create a new/ one where the parent widget 
	// is defined.
  
	delete this->ProgressDialog;

	this->ProgressDialog = new QProgressDialog("", NULL, 0, 100, parent);
	this->ProgressDialog->setRange(0, 100);
	this->ProgressDialog->setMinimumDuration(10);
}


//---------------------------[ GetParentWidget ]---------------------------\\

QWidget * QVTKProgressCommand::GetParentWidget()
{
	return this->ProgressDialog->parentWidget();
}


//-------------------------------[ setLabel ]------------------------------\\

void QVTKProgressCommand::setLabel(QString rLabel)
{
	this->label = rLabel;
}


//-------------------------------[ setTitle ]------------------------------\\

void QVTKProgressCommand::setTitle(QString rTitle)
{
	this->title = rTitle;
}


} // namespace bmia
