#ifndef __vtkParameterizePolyData_h
#define __vtkParameterizePolyData_h

#include <vtkObject.h>
#include <vtkPolyData.h>

class vtkParameterizePolyData : public vtkObject
{
public:

	static vtkParameterizePolyData * New();
	vtkTypeRevisionMacro( vtkParameterizePolyData, vtkObject );

	void SetInput( vtkPolyData * input );
	vtkPolyData * GetInput();

	void SetCentroid( double centroid[3] );
	double GetCentroidX();
	double GetCentroidY();
	double GetCentroidZ();

	void Execute();

	vtkPolyData * GetOutput();

protected:

	vtkParameterizePolyData();
	virtual ~vtkParameterizePolyData();

private:

	void NormalizeVector( double V[3] );

	vtkPolyData * Input;
	double Centroid[3];
};

#endif
