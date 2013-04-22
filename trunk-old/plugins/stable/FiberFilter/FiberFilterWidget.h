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
 * FiberFilterWidget.h
 *
 * 2010-10-05	Tim Peeters
 * - First version
 *
 * 2010-11-08	Evert van Aart
 * - Added filtering functionality
 *
 * 2010-11-22	Evert van Aart
 * - Fixed output name formatting
 *
 */


#ifndef bmia_FiberFilterWidget_h
#define bmia_FiberFilterWidget_h


/** Includes - Main Header */
#include "DTITool.h"

/** Includes - GUI */

#include "ui_fiberfilter.h"

/** Includes - Custom Files */

#include "vtk2DROIFiberFilter.h"
#include "FiberFilterPlugin.h"

/** Includes - Qt */

#include <QtDebug>

/** Includes - VTK */

#include <vtkMatrix4x4.h>


namespace bmia {


/** Forward Class Decleration */

class FiberFilterPlugin;

/** Qt widget for defining the filter ROIs for fibers. A number of these
	widgets can be placed next to each other using tabs. Each widget 
	contains a combo box for selecting an input fiber set, five combo 
	boxes for selecting up to five filter ROIs, and a line edit for
	changing the output name.
 */

class FiberFilterWidget : public QWidget, private Ui::FiberFilterForm
{
	Q_OBJECT

	public:

		/** Constructor */

		FiberFilterWidget(FiberFilterPlugin * ffp, int rFilterID);

		/** Destructor */
    
		virtual ~FiberFilterWidget();

		/** Called when a new fiber data set has been added to the plugin.
			@param ds		New fiber data set. */

		void fibersAdded(data::DataSet * ds);
		
		/** Called when an existing fiber set has changed. Since the widget
			does not keep track of data pointers (which it instead requests
			from the parent plugin when filtering), all we do is rename the
			element in the input combo box. 
			@param index	Index of changed fibers in the combo box.
			@param newName	New name of the fiber data set. */

		void fibersChanged(int index, QString newName);

		/** Called when an existing fiber set is removed. Removes the corresponding
			item from the combo box. 
			@param index	Index of deleted fibers in the combo box. */

		void fibersRemoved(int index);

		/** Called when a new Region of Interest has been added to the plugin. 
			@param ds		New ROI data set. */

		void roiAdded(data::DataSet * ds);

		/** Called when an existing ROI data set has changed. Since the widget
			does not keep track of data pointers (which it instead requests
			from the parent plugin when filtering), all we do is rename the
			elements in the ROI combo boxes. 
			@param index	Index of changed ROI in the combo boxes.
			@param newName	New name of the ROI data set. */

		void roiChanged(int index, QString newName);

		/** Called when an existing ROI data set is removed. Removes the
			corresponding items from the ROI combo boxes. 
			@param index	Index of deleted ROI in the combo boxes. */

    	void roiRemoved(int index);

	protected slots:

		/** Called when the user clicks the "Update" button. Computes the 
			output fibers for the current configuration of ROIs. */

		void update();

		/** Enable or disable GUI controls, based on the settings of 
			other controls. */

		void enableControls();

		/** Set the output name of the filter to a default name, if and
			only if the "outputNameModified" variable is set to false. */

		void setOutputName();

		/** Sets the "outputNameModified" boolean to false if the "nameEdit"
			dialog is empty, and to true otherwise. */
	
		void nameChanged();

	private:

		/** Pointer to parent plugin. */

		FiberFilterPlugin * plugin;

		/** ID of the widget, same as the tab number in the GUI. */
		int filterID;

		/** Set to true as soon as the user manually changes the text in the 
			"nameEdit" dialog. If it is false, and the user changes the input
			data set, the output name is set to default, which is of the format
			"<inputName> - Filter <filterId>". If it is true, the existing name
			(set by the user) is not changed. */

		bool outputNameModified;

}; // class FiberFilterWidget


} // namespace bmia


#endif // bmia_FiberFilterWidget_h
