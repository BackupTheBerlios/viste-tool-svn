#include <vtkROIWidget.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkBorderRepresentation.h>
#include <vtkFiberConfidenceMapper.h>
#include <vtkCoordinate.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>

vtkCxxRevisionMacro( vtkROIWidget, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkROIWidget );

///////////////////////////////////////////////////////////////////////////
vtkROIWidget::vtkROIWidget() : vtkBorderWidget()
{
	for( int i = 0; i < 4; ++i )
		this->ROI[i] = 0;

	this->Rep = static_cast< vtkBorderRepresentation *>(
			this->GetRepresentation() );
	this->Mapper = 0;
	this->MovingOrResizing = false;
	this->SetEnabled( 0 );
	this->SetResizable( 1 );
	this->SetSelectable( 0 );

	this->ProcessEventsOff();
}

///////////////////////////////////////////////////////////////////////////
vtkROIWidget::~vtkROIWidget()
{
}

///////////////////////////////////////////////////////////////////////////
int vtkROIWidget::SubclassSelectAction()
{
	this->MovingOrResizing = true;
	return 0;
}

///////////////////////////////////////////////////////////////////////////
int vtkROIWidget::SubclassEndSelectAction()
{
	this->MovingOrResizing = false;
	return 0;
}

///////////////////////////////////////////////////////////////////////////
int vtkROIWidget::SubclassTranslateAction()
{
	this->MovingOrResizing = true;
	return 0;
}

///////////////////////////////////////////////////////////////////////////
int vtkROIWidget::SubclassMoveAction()
{
	if( this->MovingOrResizing )
	{
		if( this->Mapper == 0 )
		{
			//std::cout << "vtkROIWidget::SubclassMoveAction() " \
			//		"mapper is NULL" << std::endl;
			return 0;
		}

		vtkCoordinate * p1 = this->Rep->GetPositionCoordinate();
		vtkCoordinate * p2 = this->Rep->GetPosition2Coordinate();

		vtkRenderer * viewport = this->GetInteractor()->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
		int * dp1 = p1->GetComputedDisplayValue( viewport );
		int * dp2 = p2->GetComputedDisplayValue( viewport );

		this->ROI[0] = dp1[0];
		this->ROI[1] = dp1[1];
		this->ROI[2] = abs( dp2[0] - dp1[0] );
		this->ROI[3] = abs( dp2[1] - dp1[1] );

		this->Mapper->SetROI( this->ROI );
	}

	return 0;
}

///////////////////////////////////////////////////////////////////////////
void vtkROIWidget::SetMapper( vtkFiberConfidenceMapper * mapper )
{
	this->Mapper = mapper;
}

///////////////////////////////////////////////////////////////////////////
vtkFiberConfidenceMapper * vtkROIWidget::GetMapper()
{
	return this->Mapper;
}