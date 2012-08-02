/*
 * TransformationPlugin.h
 *
 * 2011-04-27	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 */


#ifndef bmia_TransformationPlugin_h
#define bmia_TransformationPlugin_h


/** Define the UI class */

namespace Ui 
{
	class TransformationForm;
}

/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_Transformation.h"

/** Includes - Qt */

#include <QList>
#include <QProgressDialog>

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>

/** Includes - Custom Files */

#include "TensorMath/vtkTensorMath.h"


namespace bmia {


class TransformationPlugin : 	public plugin::Plugin,
								public data::Consumer,
								public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Current Version */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

		/** Constructor */

		TransformationPlugin();

		/** Destructor */

		~TransformationPlugin();

		/** Initialize the plugin. */

		void init();

		/** Returns the Qt widget that gives the user control. This 
			implements the GUI interface. */
    
		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed.
			@param ds	Modified data set. */
   
		void dataSetRemoved(data::DataSet * ds);

		enum ImageType
		{
			IT_DTI = 0,
			IT_Scalars
		};

	protected slots:

		void flipImageX()	{		flipImage(0);		}
		void flipImageY()	{		flipImage(1);		}
		void flipImageZ()	{		flipImage(2);		}

	private:

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::TransformationForm * ui;

		QList<data::DataSet *> imageList;

		void flipImage(int axis);

}; // class TransformationPlugin


} // namespace bmia


#endif // bmia_TransformationPlugin_h
