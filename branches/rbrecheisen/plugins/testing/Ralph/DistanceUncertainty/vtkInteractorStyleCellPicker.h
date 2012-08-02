#ifndef __vtkInteractorStyleCellPicker_h
#define __vtkInteractorStyleCellPicker_h

#include <vtkActor.h>
#include <vtkCommand.h>
#include <vtkRenderer.h>
#include <vtkInteractorStyleTrackballCamera.h>

class vtkInteractorStyleCellPicker : public vtkInteractorStyleTrackballCamera
{
public:

	static vtkInteractorStyleCellPicker * New();
	vtkTypeRevisionMacro( vtkInteractorStyleCellPicker, vtkInteractorStyleTrackballCamera );

	void SetRenderer( vtkRenderer * renderer );
	vtkRenderer * GetRenderer();

	void SetPickActorToDefault();
	void SetPickActor( vtkActor * actor );
	vtkActor * GetPickActor();

	void SetEventHandler( vtkCommand * handler );
	vtkCommand * GetEventHandler();

	double * GetPickedPosition();
	void GetPickedPosition( double position[3] );

	virtual void OnLeftButtonDown();

protected:

	vtkInteractorStyleCellPicker();
	virtual ~vtkInteractorStyleCellPicker();

private:

	vtkActor * CreateDefaultPickActor();

	vtkRenderer * Renderer;
	vtkCommand * EventHandler;
	vtkActor * PickActor;

	double PickedPosition[3];
};

#endif
