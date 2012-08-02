// Includes DTITool

#include <DistanceUncertaintyPlugin.h>
#include <core/Core.h>

// Includes VTK

#include <vtkActor.h>

namespace bmia
{
	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin::DistanceUncertaintyPlugin() :
		plugin::Plugin( "DistanceUncertaintyPlugin" ),
		plugin::Visualization(),
		data::Consumer(),
		plugin::GUI()
	{
		// Create QT widget that will hold plugin's GUI
		_widget = new QWidget;

		// Create VTK prop for this plugin
		_prop = vtkActor::New();
	}

	///////////////////////////////////////////////////////////////////////////
	DistanceUncertaintyPlugin::~DistanceUncertaintyPlugin()
	{
		// Delete QT objects
		delete _widget;

		// Delete VTK objects
		_prop->Delete();
		_prop = 0;
	}

	///////////////////////////////////////////////////////////////////////////
	vtkProp * DistanceUncertaintyPlugin::getVtkProp()
	{
		return _prop;
	}

	///////////////////////////////////////////////////////////////////////////
	QWidget * DistanceUncertaintyPlugin::getGUI()
	{
		return _widget;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::dataSetAdded( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::dataSetRemoved( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;
	}

	///////////////////////////////////////////////////////////////////////////
	void DistanceUncertaintyPlugin::dataSetChanged( data::DataSet * dataset )
	{
		// Check if dataset is not NULL
		if( dataset == 0 )
			return;
	}
}

Q_EXPORT_PLUGIN2( libDistanceUncertaintyPlugin, bmia::DistanceUncertaintyPlugin )
