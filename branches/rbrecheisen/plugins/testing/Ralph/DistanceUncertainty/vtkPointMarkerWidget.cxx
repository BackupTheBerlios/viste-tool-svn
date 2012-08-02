#include <vtkPointMarkerWidget.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkActor.h>
#include <vtkObjectFactory.h>

vtkCxxRevisionMacro( vtkPointMarkerWidget, "$Revision: 1.0 $" );
vtkStandardNewMacro( vtkPointMarkerWidget );

//////////////////////////////////////////////////////////////////////
vtkPointMarkerWidget::vtkPointMarkerWidget()
{
	this->Position[0] = 0.0;
	this->Position[1] = 0.0;
	this->Position[2] = 0.0;

	this->Color[0] = 0.0;
	this->Color[1] = 0.0;
	this->Color[2] = 1.0;

	this->Size = 2.0;
}

//////////////////////////////////////////////////////////////////////
vtkPointMarkerWidget::~vtkPointMarkerWidget()
{
}

//////////////////////////////////////////////////////////////////////
void vtkPointMarkerWidget::UpdateGeometry()
{
	vtkSphereSource * sphere = vtkSphereSource::New();
	sphere->SetThetaResolution( 64 );
	sphere->SetPhiResolution( 32 );
	sphere->SetRadius( this->Size );
	sphere->SetCenter( this->Position );

	vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
	mapper->SetInput( sphere->GetOutput() );
	sphere->Delete();

	vtkActor * actor = vtkActor::New();
	actor->SetMapper( mapper );
	actor->GetProperty()->SetColor( this->Color[0], this->Color[1], this->Color[2] );
	mapper->Delete();

	this->AddPart( actor );
	actor->Delete();
}

//////////////////////////////////////////////////////////////////////
void vtkPointMarkerWidget::SetSize( double size )
{
	this->Size = size;
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetSize()
{
	return this->Size;
}

//////////////////////////////////////////////////////////////////////
void vtkPointMarkerWidget::SetPosition( double position[3] )
{
	this->Position[0] = position[0];
	this->Position[1] = position[1];
	this->Position[2] = position[2];
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetPositionX()
{
	return this->Position[0];
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetPositionY()
{
	return this->Position[1];
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetPositionZ()
{
	return this->Position[2];
}

//////////////////////////////////////////////////////////////////////
void vtkPointMarkerWidget::SetColor( double r, double g, double b )
{
	this->Color[0] = r;
	this->Color[1] = g;
	this->Color[2] = b;
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetColorR()
{
	return this->Color[0];
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetColorG()
{
	return this->Color[1];
}

//////////////////////////////////////////////////////////////////////
double vtkPointMarkerWidget::GetColorB()
{
	return this->Color[2];
}
