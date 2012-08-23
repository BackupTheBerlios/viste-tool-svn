#ifndef __vtkDistanceArrowWidget_h
#define __vtkDistanceArrowWidget_h

#include <vtkPropAssembly.h>

class vtkDistanceArrowWidget : public vtkPropAssembly
{
public:

	static vtkDistanceArrowWidget * New();
	vtkTypeRevisionMacro( vtkDistanceArrowWidget, vtkPropAssembly );

	void SetStartPoint( double P[3] );
	double GetStartPointX();
	double GetStartPointY();
	double GetStartPointZ();

	void SetEndPoint( double P[3] );
	double GetEndPointX();
	double GetEndPointY();
	double GetEndPointZ();

	void SetDistance( double distance );
	double GetDistance();

	void UpdateGeometry();

protected:

	vtkDistanceArrowWidget();
	virtual ~vtkDistanceArrowWidget();

private:

	double StartPoint[3];
	double EndPoint[3];

	double Distance;
};

#endif
