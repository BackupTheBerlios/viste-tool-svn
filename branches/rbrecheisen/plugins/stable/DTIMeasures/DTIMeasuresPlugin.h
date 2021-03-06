/*
 * DTIMeasuresPlugin.h
 *
 * 2010-03-09	Tim Peeters
 * - First version
 *
 * 2011-01-20	Evert van Aart
 * - Added support for transformation matrices.
 * - Anisotropy images are now computed on demand.
 *
 * 2011-03-10	Evert van Aart
 * - Version 1.0.0.
 * - Increased stability when changing or removing data sets.
 * - Added additional comments.
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 */


#ifndef bmia_DTIMeasures_DTIMeasuresPlugin_h
#define bmia_DTIMeasures_DTIMeasuresPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "data/DataSet.h"
#include "TensorMath/AnisotropyMeasures.h"
#include "vtkTensorToEigensystemFilter.h"
#include "vtkEigenvaluesToAnisotropyFilter.h"

/** Includes - Qt */

#include <QDebug>

/** Includes - VTK */

#include <vtkExecutive.h>
#include <vtkMatrix4x4.h>


namespace bmia {


/** This plugin computes the eigensystem data of DTI tensors (eigenvalues and
	eigenvectors), and, from that, several different anisotropy images. 
	If the input image contains a transformation matrix, it is copied to the 
	output. To save memory, the anisotropy measure images are computed by 
	request only. This means that the "vtkImageData" object passed to the 
	output data set is empty at first, and will only be computed when its 
	"Update" function is called.
*/

class DTIMeasuresPlugin :	public plugin::Plugin, 
							public data::Consumer
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.0.2";
		}

		/** Constructor */

		DTIMeasuresPlugin();

		/** Destructor */

		~DTIMeasuresPlugin();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		virtual void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		virtual void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed. */

		virtual void dataSetRemoved(data::DataSet * ds);

	protected:

	
	private:

		/** Structure to keep track of all outputs for a specific input image. */

		struct OutputInformation
		{
			data::DataSet * input;
			data::DataSet * eigenOutput;
			QList<data::DataSet *> aiOutputs;
		};

		/** List of all input images that have been handled by this plugin. */

		QList<OutputInformation> images;

		/** General list of all anisotropy filters that were created by this plugin. */

		QList<vtkEigenvaluesToAnisotropyFilter *> aiFilters;

		/** Find a specified data set pointer in the "images" list. On success,
			the index of the image is copied to "imageID", and true is returned.
			If the list does not contain the specified pointer, false is returned.
			@param ds		Target data set.
			@param imageID	Output index of the data set. */

		bool findInputImage(data::DataSet * ds, int& imageID);

		/** Find a specified eigensystem data set pointer in the "images" list. 
			On success, the index of the image is copied to "imageID", and true 
			is returned. If the list does not contain the specified pointer, 
			false is returned.
			@param ds		Target data set.
			@param imageID	Output index of the data set. */

		bool findEigenImage(data::DataSet * ds, int& imageID);

		/** Find a specified anisotropy data set pointer in the "images" list. 
			On success, the index of the image is copied to "imageID", the index
			of the AI image within the "aiOutputs" list is copied to "aiID", 
			and true is returned. If the list does not contain the specified 
			pointer, false is returned.
			@param ds		Target data set.
			@param imageID	Index of the input used to generate the target AI image. 
			@param aiID		Index of the target AI image within the "aiOutputs" list. */

		bool findAIImage(data::DataSet * ds, int& imageID, int& aiID);


}; // class DTIMeasuresPlugin


} // namespace bmia


#endif // bmia_DTIMeasures_DTIMeasuresPlugin_h
