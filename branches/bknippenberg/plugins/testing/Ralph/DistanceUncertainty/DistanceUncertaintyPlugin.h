#ifndef bmia_DistanceUncertainty_DistanceUncertaintyPlugin_h
#define bmia_DistanceUncertainty_DistanceUncertaintyPlugin_h

#include "DTITool.h"

class vtkActor;

//! \class DistanceUncertaintyPlugin
//! \file  DistanceUncertaintyPlugin.h
//! \brief This plugin allows the exploration of distance measurements taking into
//! account uncertainty in the spatial extent of the structures of interest.

namespace bmia
{
	class DistanceUncertaintyPlugin :	public plugin::Plugin,
										public plugin::Visualization,
										public data::Consumer,
										public plugin::GUI
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::Visualization )
		Q_INTERFACES( bmia::data::Consumer )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		//! Constructor
		DistanceUncertaintyPlugin();

		//! Destructor
		virtual ~DistanceUncertaintyPlugin();

		//! Returns plugin's VTK prop
		vtkProp * getVtkProp();

		//! Returns plugin's QT widget
		QWidget * getGUI();

		//! Handle dataset events
		void dataSetAdded  ( data::DataSet * dataset );
		void dataSetRemoved( data::DataSet * dataset );
		void dataSetChanged( data::DataSet * dataset );

	private:

		QWidget		* _widget;		// The QT widget holding this plugin's GUI
		vtkActor	* _prop;		// The VTK prop of this plugin
	};
}

#endif
