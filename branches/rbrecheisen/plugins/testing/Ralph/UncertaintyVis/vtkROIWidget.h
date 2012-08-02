#ifndef __vtkROIWidget_h
#define __vtkROIWidget_h

#include <vtkBorderWidget.h>

class vtkBorderRepresentation;
class vtkFiberConfidenceMapper;

class vtkROIWidget : public vtkBorderWidget
{
public:

	static vtkROIWidget * New();
	vtkTypeRevisionMacro( vtkROIWidget, vtkBorderWidget );

	void SetMapper( vtkFiberConfidenceMapper * mapper );
	vtkFiberConfidenceMapper * GetMapper();

protected:

	vtkROIWidget();
	virtual ~vtkROIWidget();

	int SubclassMoveAction();
	int SubclassSelectAction();
	int SubclassEndSelectAction();
	int SubclassTranslateAction();

	vtkBorderRepresentation * Rep;
	vtkFiberConfidenceMapper * Mapper;

	int ROI[4];

	bool MovingOrResizing;
};

#endif
