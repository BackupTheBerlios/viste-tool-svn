/*
 * vtkTextProgressCommand.h
 *
 * 2009-11-28	Tim Peeters
 * - First version
 */

#ifndef bmia_core_vtkTextProgressCommand_h
#define bmia_core_vtkTextProgressCommand_h

#include <vtkCommand.h>

namespace bmia {

/**
 * vtkCommand that can be used to monitor progress events.
 * These events are then displayed via cout.
 * This is a drop-in class to output updates to cout until
 * I have a proper GUI where I can show a progress dialog.
 */
class vtkTextProgressCommand : public vtkCommand
{
public:
    static vtkTextProgressCommand* New();
    virtual void Execute(vtkObject* caller, unsigned long eventId, void* callData);

protected:		
    vtkTextProgressCommand();
    ~vtkTextProgressCommand();

}; // class vtkTextProgressCommand
} // namespace bmia
#endif // bmia_core_vtkTextProgrerssCommand_h
