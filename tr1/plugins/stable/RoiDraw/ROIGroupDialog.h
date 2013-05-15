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
 * ROIGroupDialog.h
 *
 * 2011-02-16	Evert van Aart
 * - First version. 
 * 
 */

#ifndef bmia_RoiDrawPlugin_ROIGroupDialog_h
#define bmia_RoiDrawPlugin_ROIGroupDialog_h


/** Includes - Qt */

#include <QDialog>
#include <QList>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QSizePolicy>


namespace bmia {


/** This class represents a dialog box, which allows the user to group together
	multiple ROIs. It is created and maintained by the ROI Edit plugin. Grouping
	the ROIs consists of three steps: Choosing a name (or use a default name);
	Selecting the input ROIs; and selecting whether or not these input ROIs
	should be deleted after grouping them. 
*/

class ROIGroupDialog : public QDialog
{
	public:

		/** Constructor */

		ROIGroupDialog();

		/** Destructor */

		~ROIGroupDialog();

		/** Set the default name for the group, which will be entered in the
			line edit widget of the first step
			@param defaultName	Default name for the new group. */

		void setDefaultName(QString defaultName);

		/** Add the name of an existing ROI to the table widget.
			@param newName		Name of an existing ROI. */

		void addROIName(QString newName);

		/** Return the name for the group chosen by the user. */

		QString getGroupName();

		/** Return whether or not the input ROIs should be deleted. */

		bool getDeleteInputROIs();

		/** Return the indices of the ROIs selected by the user. */

		QList<int> getSelectedROIs();

	private:

		/** Label for the first step (choosing the group name). */

		QLabel * step1Label;

		/** Line edit used in the first step (choosing the group name). */

		QLineEdit * step1LineEdit;

		/** Label for the second step (selecting the ROIs). */

		QLabel * step2Label;

		/** Table widget used in the second step. Contains a single checkbox per
			input ROI. The user can use this table to select the desired ROIs. */

		QTableWidget * step2TableWidget;

		/** Label for the last step (deleting input ROIs). */

		QLabel * step3Label;

		/** Checkbox for the last step (deleting input ROIs). When checked, the 
			checked input ROIs will be deleted after merging them. */

		QCheckBox * step3Checkbox;

		/** OK button. */

		QPushButton * okButton;

		/** Cancel button. */

		QPushButton * cancelButton;

		/** Horizontal layout for the buttons. */

		QHBoxLayout * buttonLayout;

		/** Vertical layout for the dialog window. */

		QVBoxLayout * mainLayout;

}; // class ROIGroupDialog


} // namespace bmia


#endif // bmia_RoiDrawPlugin_ROIGroupDialog_h