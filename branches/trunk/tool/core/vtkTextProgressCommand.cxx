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
 * vtkTextProgressCommand.cxx
 *
 * 2009-11-28	Tim Peeters
 * -First version.
 */

#include "vtkTextProgressCommand.h"
//#include "Core.h"
#include "UserOutput.h"
#include <vtkAlgorithm.h>
#include <QString>

namespace bmia {

vtkTextProgressCommand* vtkTextProgressCommand::New()
{
    return new vtkTextProgressCommand;
}

vtkTextProgressCommand::vtkTextProgressCommand()
{
    // nothing to do.
}

vtkTextProgressCommand::~vtkTextProgressCommand()
{
    // nothing to destroy.
}

void vtkTextProgressCommand::Execute(vtkObject* caller, unsigned long eventId, void* callData)
{
  // eventId is ignored. It should always be a ProgressEvent.

  double progress = *(static_cast<double*>(callData)) * 100.0;
  vtkAlgorithm* alg = vtkAlgorithm::SafeDownCast(caller);
  // if alg == NULL, then caller class is not a subclass of vtkAlgorithm

  QString progressText;
  if (alg)
    { // caller was a vtkAlgorithm
    if (alg->GetProgressText())
      { // progress text was set by the vtkAlgorithm
      progressText.append(alg->GetProgressText());
      }
    else
      { // no progress text
      progressText.append("Progress for algorithm ");
      progressText.append(caller->GetClassName());
      }
    } // if
  else
    { // caller was not a vtkAlgorithm
    progressText.append("Progress for VTK object ");
    progressText.append(caller->GetClassName());
    }
  alg = NULL;
  //this->ProgressDialog->setLabelText(progressText);
  //this->ProgressDialog->setValue((int)progress);

  QTextStream out(stdout);
  out<<"====== progress text = "<<progressText<<endl;
  out<<"====== progress = "<<progress<<endl;

  // make sure the progress dialog is visible:
  // commented out. It is not really necessary if minimumDuration is
  // small enough.
  //if (progress < 100.0) this->ProgressDialog->show();
  //qApp->processEvents();
}

} // namespace bmia
