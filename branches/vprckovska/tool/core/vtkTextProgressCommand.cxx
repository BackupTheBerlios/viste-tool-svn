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
