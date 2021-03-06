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
 * vtkTextProgressCommand.h
 *
 * 2009-11-28	Tim Peeters
 * - First version
 */


#ifndef bmia_core_vtkTextProgressCommand_h
#define bmia_core_vtkTextProgressCommand_h


/** Includes - VTK */

#include <vtkCommand.h>


namespace bmia {


/** This class that can be used to monitor progress events. These events are 
	then displayed via the standard output. This class is not used anymore,
	the progress bar dialog of "QVTKProgressCommand" is used instead. */

class vtkTextProgressCommand : public vtkCommand
{
	public:

		/** Constructor Call. */

		static vtkTextProgressCommand * New();

		/** Called when the caller object updates its progress.
			@param caller		Algorithm that fired the progress event.
			@param eventId		Event identifier.
			@param callData		New progress value. */

		virtual void Execute(vtkObject * caller, unsigned long eventId, void * callData);

	protected:

		/** Constructor */

		vtkTextProgressCommand();

		/** Destructor */

		~vtkTextProgressCommand();

}; // class vtkTextProgressCommand


} // namespace bmia


#endif // bmia_core_vtkTextProgrerssCommand_h
