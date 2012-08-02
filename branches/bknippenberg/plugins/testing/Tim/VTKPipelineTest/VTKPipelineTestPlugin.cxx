/*
 * VTKPipelineTestPlugin.cxx
 *
 * 2010-03-09	Tim Peeters
 * - First version
 */

#include "VTKPipelineTestPlugin.h"
#include "data/DataSet.h"

#include <QDebug>
#include <vtkSimpleImageFilterExample.h>
#include <vtkExecutive.h>
#include <vtkAlgorithmOutput.h>

namespace bmia {

VTKPipelineTestPlugin::VTKPipelineTestPlugin() : plugin::Plugin("VTK pipeline test")
{
    // nothing to do :)
}

VTKPipelineTestPlugin::~VTKPipelineTestPlugin()
{
    // nothing to destroy. How peaceful.
}

void VTKPipelineTestPlugin::dataSetAdded(data::DataSet* ds)
{
	if (ds->getKind() != "DTI") return;

	vtkImageData* dti = ds->getVtkImageData();
	Q_ASSERT(dti);

    qDebug()<<"input information should be"<<dti->GetProducerPort()->GetProducer()->GetExecutive()->GetOutputInformation(0);

	QString basename = ds->getName();
	qDebug()<<"Dataset"<<ds<<"with name"<<basename<<"and kind"<<ds->getKind()<<"seems usable.";

	vtkSimpleImageFilterExample* filter = vtkSimpleImageFilterExample::New();
	qDebug()<<"setting intput to"<<dti;
	filter->DebugOn();
	qDebug()<<"Executive 1 = "<<filter->GetExecutive();
	filter->SetInput(dti);
	qDebug()<<"input is"<<filter->GetInput();
	qDebug()<<"total input connections ="<<filter->GetTotalNumberOfInputConnections();
	qDebug()<<"#input connections on port 0 ="<<filter->GetNumberOfInputConnections(0);
	qDebug()<<"executive 2 = "<<filter->GetExecutive();
	qDebug()<<"producer port = "<<dti->GetProducerPort();
	filter->SetInputConnection(0,dti->GetProducerPort());
	qDebug()<<"total input connections ="<<filter->GetTotalNumberOfInputConnections();
	qDebug()<<"#input connections on port 0 ="<<filter->GetNumberOfInputConnections(0);
	qDebug()<<"input connection ="<<filter->GetInputConnection(0,0);
	qDebug()<<"Executive 3 = "<<filter->GetExecutive();

	vtkExecutive* executive = filter->GetExecutive();
	qDebug()<<"executive ="<<executive;
	qDebug()<<"executive input ports = "<<executive->GetNumberOfInputPorts();
	qDebug()<<"exec input data = "<<executive->GetInputData(0,0);
}

void VTKPipelineTestPlugin::dataSetChanged(data::DataSet* ds)
{
    // do nothing for now.
    // The input data is read from file and should not change.
}

void VTKPipelineTestPlugin::dataSetRemoved(data::DataSet* ds)
{
    // nothing to do.
}

} // namespace bmia
Q_EXPORT_PLUGIN2(libVTKPipelineTestPlugin, bmia::VTKPipelineTestPlugin)
