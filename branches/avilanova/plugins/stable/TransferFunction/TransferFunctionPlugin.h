/*
 * TransferFunctionPlugin
 *
 * 2010-02-22	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-04	Evert van Aart
 * - Refactored code, added comments
 * - Changed the GUI.
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.0.0.
 * - Prevented divide-by-zero errors for scalar images with zero range. 
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.1.
 * - When saving transfer functions, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 */


#ifndef bmia_TransferFunctionPlugin_h
#define bmia_TransferFunctionPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes = Custom Files */

#include "QTransferFunctionCanvas.h"

/** Includes - GUI */

#include "ui_TransferFunctionForm.h"

/** Includes - VTK */

#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkPointData.h>

/** Includes - Qt */

#include <QtGui>
#include <QDir>


/** Define the UI class */

namespace Ui
{
    class TransferFunctionForm;
}


namespace bmia {



/** A GUI plugin that allows the user to edit transfer functions. The plugin produces data sets
	of type "transfer function" when the user presses the "New" button, and also consumes data sets 
	of this type added through some other means (such as loading ".tf" files). The transfer function
	data sets contain the minimum and maximum of the scalar range as attributes; their attributes
	may also contain a piecewise function object. The plugin can use data sets of type "scalar
	volume" to automatically determine the scalar range for a transfer function. 
 */


class TransferFunctionPlugin : public plugin::Plugin, public plugin::GUI, public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::GUI)
    Q_INTERFACES(bmia::data::Consumer);

	public:

		/** Return current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */
		
		TransferFunctionPlugin();

		/** Destructor */
    
		~TransferFunctionPlugin();

		/** Return the widget that is shown in the GUI. */
		
		QWidget * getGUI();

		/** This function is called when a new data set becomes available.
			@param ds	The new data set that was added. */
    
		void dataSetAdded(data::DataSet * ds);

		/** This function is called when an already available data set was changed.
			@param ds	The data set that has been updated. */
    
		void dataSetChanged(data::DataSet * ds);

		/** This function is called when a data set that was available has been removed.
			@param ds	The data set that was removed from the pool of data sets. */
    
		void dataSetRemoved(data::DataSet * ds);

	protected:
    
		/** The widget shown in the GUI. */

		QWidget * qWidget;

		/** The form defining the GUI. */

		Ui::TransferFunctionForm * ui;

		/** Lists of available data sets of type "scalar volume". */
    
		QList<data::DataSet *> compatibleDataSets;

		/** Lists of available data sets of type "transfer function". */

		QList<data::DataSet *> compatibleTransferFunctions;

		/** List of available transfer functions. */

		QList<vtkColorTransferFunction *> transferFunctions;

		/** List of available piecewise functions. */
    
		QList<vtkPiecewiseFunction *> piecewiseFunctions;

		/** Save the transfer function to a file.
			@param pTf	Pointer to the transfer function, cannot be NULL.
			@param pPf	Pointer to the piecewise function, can be NULL. */

		bool saveTransferFunction(vtkColorTransferFunction * pTf, vtkPiecewiseFunction * pPf);


	private slots:

		/** Toggle histogram flattening on or off.
			@param checked	Whether the flattening checkbox is checked or not. */
    
		void flatteningToggled(bool checked);

		/** Called when a colormap is selected.
			@param index	Index of the colormap. */
     
		void setCurrentColormap(int index);

		/** Called when an image data set is selected.
			@param index	Index of the dataset. */

		void setCurrentDataset(int index);

		/** Called to make the range of the transfer function equal to the range of 
			the selected image data set. */
    
		void adoptRange();

		/** Called when the range is changed using the spin boxes. */
    
		void setRange();

		/** Called when the "Save" button is pressed. */
    
		void save();

		/** Called when something is changed to the transfer function. */
    
		void transferFunctionChanged();

		/** Add a new transfer function to the data sets. */
    
		void addNew();

		/** Add new piecewise function to the current transfer function (only if 
			the transfer function does not already have a piecewise function). */
    
	    void addPiecewisefunction();

}; // class TransferFunctionPlugin


} // namespace bmia


#endif // bmia_TransferFunctionPlugin_h
