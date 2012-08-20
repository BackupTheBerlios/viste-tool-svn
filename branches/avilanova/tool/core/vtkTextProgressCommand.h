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
