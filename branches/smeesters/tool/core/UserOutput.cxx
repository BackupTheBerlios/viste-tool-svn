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
 * UserOutput.cxx
 *
 * 2009-11-26	Tim Peeters
 * - First version.
 *
 * 2011-04-26	Evert van Aart
 * - The function "showMessage" now creates a message box.
 * - Improved the way progress bars are handled. Each algorithm gets a separate
 *   progress bar, which will exist until the plugin that created it destroys it.
 *   This way, it's easier to maintain progress bars for algorithms that update
 *   through the VTK pipeline; the algorithm's progress bar will only appear when
 *   the algorithm updates, and will remain hidden the rest of the time.
 * - Tidied up the code.
 *
 */


/** Includes */

#include "UserOutput.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

UserOutput::UserOutput() : outStream(stdout)
{

}


//------------------------------[ Destructor ]-----------------------------\\

UserOutput::~UserOutput()
{
	// Flush the text stream
	this->outStream.flush();
	
	QHash<vtkAlgorithm *, QVTKProgressCommand *>::iterator i;

	// Loop through all registered algorithms
	for (i = this->registeredAlgorithms.begin(); i != this->registeredAlgorithms.end(); ++i)
	{
		// Delete the progress command
		QVTKProgressCommand * progressCommand = i.value();
		vtkAlgorithm * algorithm = i.key();

		algorithm->RemoveObserver(progressCommand);
		progressCommand->Delete();
	}

	// Clear the hash map of registered algorithms
	this->registeredAlgorithms.clear();
}


//-----------------------------[ showMessage ]-----------------------------\\

void UserOutput::showMessage(QString msg, QString title, bool logMsg)
{
	// If the "title" string is empty, we simply use the default string
	QString dialogTitle = title.isEmpty() ? QString("DTITool") : title;

	// Show the message box
	QMessageBox::warning(NULL, dialogTitle, msg);

	// If requested, add the message to the log
	if (logMsg)
		this->logMessage("[Message to User] " + msg);
}


//------------------------------[ logMessage ]-----------------------------\\

void UserOutput::logMessage(QString msg)
{
	// Write the message to the text stream
    this->outStream << "LOG: " << msg << endl;
}


//--------------------[ createProgressBarForAlgorithm ]--------------------\\

void UserOutput::createProgressBarForAlgorithm(vtkAlgorithm * algorithm, QString title, QString label)
{
	// Do nothing if this algorithm already has a progress bar
	if (this->registeredAlgorithms.contains(algorithm))
		return;

	// Create a new progress command, which will monitor for progress changes
	QVTKProgressCommand * newCommand = QVTKProgressCommand::New();
	
	// Set the title and label, if they have been passed to this function
	if (title.isEmpty() == false)
		newCommand->setTitle(title);

	if (label.isEmpty() == false)
		newCommand->setLabel(label);

	// Add the progress command to the algorithm as an observer for progress events
	algorithm->AddObserver(vtkCommand::ProgressEvent, newCommand);

	// Add the pair of pointers to the hash table
	this->registeredAlgorithms.insert(algorithm, newCommand);
}


//--------------------[ deleteProgressBarForAlgorithm ]--------------------\\

void UserOutput::deleteProgressBarForAlgorithm(vtkAlgorithm * algorithm)
{
	// Do nothing if the algorithm was never registered
	if (this->registeredAlgorithms.contains(algorithm) == false)
		return;

	// Get the progress command for this algorithm
	QVTKProgressCommand * currentCommand = this->registeredAlgorithms.value(algorithm);

	// Remove the command from the algorithm, and delete it
	algorithm->RemoveObserver(currentCommand);
	currentCommand->Delete();

	// Remove the hash table entry
	this->registeredAlgorithms.remove(algorithm);
}


} // namespace bmia
