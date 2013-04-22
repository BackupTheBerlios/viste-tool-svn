#include <vtkDistanceArrowWidget.h>
#include <vtkPropCollection.h>
#include <vtkProp.h>
#include <vtkActor.h>
#include <vtkLineSource.h>
#include <vtkConeSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkCaptionActor2D.h>
#include <vtkObjectFactory.h>
#include <vtkTextActor.h>
#include <vtkProperty.h>
#include <vtkTextProperty.h>
#include <vtkTubeFilter.h>
#include <vtkMath.h>

#include <sstream>
#include <string>

vtkCxxRevisionMacro( vtkDistanceArrowWidget, "$Revision: 1.0 $" );
vtkStandardNewMacro( vtkDistanceArrowWidget );

//////////////////////////////////////////////////////////////////////
vtkDistanceArrowWidget::vtkDistanceArrowWidget()
{
	this->StartPoint[0] = 0.0;
	this->StartPoint[1] = 0.0;
	this->StartPoint[2] = 0.0;
	this->EndPoint[0] = 0.0;
	this->EndPoint[1] = 0.0;
	this->EndPoint[2] = 0.0;
	this->Distance = 0.0;
}

//////////////////////////////////////////////////////////////////////
vtkDistanceArrowWidget::~vtkDistanceArrowWidget()
{
}

//////////////////////////////////////////////////////////////////////
void vtkDistanceArrowWidget::UpdateGeometry()
{
	double coneHeight = 3.0;
	double coneRadius = 1.5;

	double V[3], normV[3];
	vtkMath::Subtract( this->EndPoint, this->StartPoint, V );
	normV[0] = V[0];
	normV[1] = V[1];
	normV[2] = V[2];
	double length = vtkMath::Norm( V );
	vtkMath::Normalize( normV );

	double Vinv[3];
	Vinv[0] = -V[0]; Vinv[1] = -V[1]; Vinv[2] = -V[2];

	double centerStart[3];
	centerStart[0] = this->StartPoint[0] + 0.5 * coneHeight * normV[0];
	centerStart[1] = this->StartPoint[1] + 0.5 * coneHeight * normV[1];
	centerStart[2] = this->StartPoint[2] + 0.5 * coneHeight * normV[2];

	vtkConeSource * coneStart = vtkConeSource::New();
	coneStart->SetHeight( coneHeight );
	coneStart->SetRadius( coneRadius );
	coneStart->SetCenter( centerStart );
	coneStart->SetResolution( 32 );
	coneStart->SetDirection( Vinv );

	double centerEnd[3];
	centerEnd[0] = this->StartPoint[0] + (length - 0.5 * coneHeight) * normV[0];
	centerEnd[1] = this->StartPoint[1] + (length - 0.5 * coneHeight) * normV[1];
	centerEnd[2] = this->StartPoint[2] + (length - 0.5 * coneHeight) * normV[2];

	vtkConeSource * coneEnd = vtkConeSource::New();
	coneEnd->SetHeight( coneHeight );
	coneEnd->SetRadius( coneRadius );
	coneEnd->SetCenter( centerEnd );
	coneEnd->SetResolution( 32 );
	coneEnd->SetDirection( V );

	vtkPolyDataMapper * mapperStart = vtkPolyDataMapper::New();
	mapperStart->SetInput( coneStart->GetOutput() );
	vtkActor * actorStart = vtkActor::New();
	actorStart->SetMapper( mapperStart );
	actorStart->GetProperty()->SetColor( 1, 1, 0 );
	this->AddPart( actorStart );

	vtkPolyDataMapper * mapperEnd = vtkPolyDataMapper::New();
	mapperEnd->SetInput( coneEnd->GetOutput() );
	vtkActor * actorEnd = vtkActor::New();
	actorEnd->SetMapper( mapperEnd );
	actorEnd->GetProperty()->SetColor( 1, 1, 0 );
	this->AddPart( actorEnd );

	vtkLineSource * line = vtkLineSource::New();
	line->SetPoint1( centerStart );
	line->SetPoint2( centerEnd );

	vtkTubeFilter * filter = vtkTubeFilter::New();
	filter->SetInput( line->GetOutput() );
	filter->SetRadius( 0.5 );
	filter->SetNumberOfSides( 16 );

	vtkPolyDataMapper * mapper = vtkPolyDataMapper::New();
	mapper->SetInput( filter->GetOutput() );

	vtkActor * actor = vtkActor::New();
	actor->SetMapper( mapper );
	actor->GetProperty()->SetColor( 1, 1, 1 );
	this->AddPart( actor );

	std::stringstream str;
	str.setf( ios::fixed, ios::floatfield );
	str.precision( 1 );
	str << this->Distance << " mm";

	vtkCaptionActor2D * caption = vtkCaptionActor2D::New();
	caption->SetCaption( str.str().c_str() );
	caption->SetPosition( 0, 0 );
	caption->SetAttachmentPoint( this->StartPoint[0], this->StartPoint[1], this->StartPoint[2] );
	caption->GetCaptionTextProperty()->SetColor( 1, 1, 1 );
	caption->GetCaptionTextProperty()->SetFontFamilyToTimes();
	caption->GetCaptionTextProperty()->SetFontSize( 16 );
	caption->GetCaptionTextProperty()->BoldOff();
	caption->GetCaptionTextProperty()->ItalicOff();
	caption->GetCaptionTextProperty()->ShadowOff();
	caption->GetTextActor()->SetTextScaleModeToNone();
	caption->BorderOff();
	caption->VisibilityOn();
	this->AddPart( caption );

	coneStart->Delete();
	coneEnd->Delete();
	mapperStart->Delete();
	mapperEnd->Delete();
	actorStart->Delete();
	actorEnd->Delete();
	line->Delete();
	filter->Delete();
	mapper->Delete();
	actor->Delete();
	caption->Delete();
}

//////////////////////////////////////////////////////////////////////
void vtkDistanceArrowWidget::SetStartPoint( double P[3] )
{
	this->StartPoint[0] = P[0];
	this->StartPoint[1] = P[1];
	this->StartPoint[2] = P[2];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetStartPointX()
{
	return this->StartPoint[0];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetStartPointY()
{
	return this->StartPoint[1];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetStartPointZ()
{
	return this->StartPoint[2];
}

//////////////////////////////////////////////////////////////////////
void vtkDistanceArrowWidget::SetEndPoint( double P[3] )
{
	this->EndPoint[0] = P[0];
	this->EndPoint[1] = P[1];
	this->EndPoint[2] = P[2];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetEndPointX()
{
	return this->EndPoint[0];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetEndPointY()
{
	return this->EndPoint[1];
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetEndPointZ()
{
	return this->EndPoint[2];
}

//////////////////////////////////////////////////////////////////////
void vtkDistanceArrowWidget::SetDistance( double distance )
{
	this->Distance = distance;
}

//////////////////////////////////////////////////////////////////////
double vtkDistanceArrowWidget::GetDistance()
{
	return this->Distance;
}
