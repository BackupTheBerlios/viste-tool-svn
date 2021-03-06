/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * TransformationPlugin.h
 *
 * 2011-04-27	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-08-22	Evert van Aart
 * - Version 1.0.1.
 * - Improved stability.
 * - Added more comments.
 * - Removed the "Cancel" button from the progress bar.
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


/** This class can transform images. Images are transformed in place, i.e., the
	existing image is overwritten. Can for example be used to flip an image.
*/

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
			return "1.0.1";
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

		/** Enumeration of all data types that can be flipped. */

		enum ImageType
		{
			IT_DTI = 0,		/**< DTI Tensors. */
			IT_Scalars		/**< Regular scalar values. */
		};

	protected slots:

		/** Flip the image over the X axis. */

		void flipImageX()	{		flipImage(0);		}

		/** Flip the image over the Y axis. */

		void flipImageY()	{		flipImage(1);		}

		/** Flip the image over the Z axis. */

		void flipImageZ()	{		flipImage(2);		}

	private:

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::TransformationForm * ui;

		/** List of all data sets containing images that can be transformed. */

		QList<data::DataSet *> imageList;

		/** Flip an image over the specified axis (i.e., mirror it along the
			image center). Scalar values are simply swapped; DTI tensors are
			flipped as well to ensure correct orientation.
			@param axis		Axis over which to flip the image. */

		void flipImage(int axis);

}; // class TransformationPlugin


} // namespace bmia


#endif // bmia_TransformationPlugin_h
