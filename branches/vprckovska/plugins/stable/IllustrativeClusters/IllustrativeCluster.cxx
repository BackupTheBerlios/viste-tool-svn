/** Includes */

#include "IllustrativeCluster.h"
#include "vtkIllustrativeFiberBundleMapper.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

IllustrativeCluster::IllustrativeCluster(const QString & name, 
										 vtkSmartPointer<vtkActor> actor, 
										 vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mapper)
										 : mName(name), mActor(actor), mMapper(mapper)
{
	// Default configuration settings
	this->mConfiguration.lineColor = QColor(156, 110, 110);
	this->mConfiguration.fillColor = QColor(240, 209, 209);

	this->mConfiguration.haloWidth = 0.4f;
	this->mConfiguration.haloDepth = 0.1f;

	this->mConfiguration.enableLighting = true;
	this->mConfiguration.phongConstants = Vector3(0.0, 0.7, 0.2);
	this->mConfiguration.specularPower = 1;
	this->mConfiguration.minLuminosity = 0;
	this->mConfiguration.maxLuminosity = 1;

	this->mConfiguration.enableCurvatureStrokes = false;
	this->mConfiguration.minStrokeWidth = 0.05f;
	this->mConfiguration.maxStrokeWidth = 0.2f;

	this->mConfiguration.enableSilhouette = true;
	this->mConfiguration.silhouetteWidth = 3;
	this->mConfiguration.contourWidth = 2;
	this->mConfiguration.depthThreshold = 10.0f;
}


//------------------------------[ Destructor ]-----------------------------\\

IllustrativeCluster::~IllustrativeCluster()
{
	// Don't do anything. The plugin itself removes the actor from the assembly,
	// which reduces its reference count to zero, and which in turn deleted
	// both the actor and its mapper.
}


//------------------------------[ SetColors ]------------------------------\\

void IllustrativeCluster::SetColors(QString lineColor, QString fillColor)
{
	// Set the new colors
	this->mConfiguration.lineColor.setNamedColor(lineColor);
	this->mConfiguration.fillColor.setNamedColor(fillColor);
}


} // namespace bmia
