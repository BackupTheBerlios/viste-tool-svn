#ifndef __vtkPointMarkerWidget_h
#define __vtkPointMarkerWidget_h

#include <vtkPropAssembly.h>

class vtkPointMarkerWidget : public vtkPropAssembly
{
public:

	static vtkPointMarkerWidget * New();
	vtkTypeRevisionMacro( vtkPointMarkerWidget, vtkPropAssembly );

	void SetPosition( double point[3] );
	double GetPositionX();
	double GetPositionY();
	double GetPositionZ();

	void SetSize( double size );
	double GetSize();

	void SetColor( double r, double g, double b );
	double GetColorR();
	double GetColorG();
	double GetColorB();

	void UpdateGeometry();

protected:

	vtkPointMarkerWidget();
	virtual ~vtkPointMarkerWidget();

private:

	double Position[3];
	double Color[3];
	double Size;
};

#endif
