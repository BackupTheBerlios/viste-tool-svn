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
 * DataDialog.h
 *
 * 2010-03-08	Tim Peeters
 * - First version.
 *
 * 2010-08-03	Tim Peeters
 * - Implement "data::Consumer functions".
 * - Replace "assert" by "Q_ASSERT".
 *
 * 2011-02-09	Evert van Aart
 * - Fixed range displaying for images.
 * 
 * 2011-03-14	Evert van Aart
 * - Added data set size to the dialog.
 * - Instead of completely rebuilding the tree widget every time any data set is
 *   added, modified, or deleted, we now only update the relevant item.
 *
 * 2011-05-13	Evert van Aart
 * - Modified attribute handling.
 *
 * 2011-07-21	Evert van Aart
 * - Added a destructor.
 *
 */


#ifndef bmia_DataDialog_h
#define bmia_DataDialog_h


/** Includes - Custom Files */

#include "data/Manager.h"
#include "data/DataSet.h"
#include "data/Attributes.h"
#include "data/Consumer.h"

/** Includes - Qt */

#include <QDialog>
#include <QPushButton>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QtDebug>
#include <QHash>

#include <QFileDialog>

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPolyData.h>


namespace bmia {


namespace gui {


/** The data dialog gives an overview of the data currently available to the data
	manager. For each data set, it also provide the user with some basic information,
	depending on the data set type and type of VTK data. The dialog acts as a 
	consumer plugin with regards to its interfaces, so it will be notified of
	all new, changed, and deleted data sets. 
*/

class DataDialog : public QDialog, public data::Consumer
{
	Q_OBJECT

	public:
	
		/** Construct a new data dialog with the given data manager.
			@param dManager		Data manager of the tool. 
			@param parent		Parent widget, usually the main window. */

		DataDialog(data::Manager * dManager, QWidget * parent = NULL);

		/** Destructor */

		~DataDialog();

		/** Called when a new data set is added to the manager.
			@param ds			New data set. */

		void dataSetAdded(data::DataSet * ds);

		/** Called when an existing data set is modified. 
			@param ds			Modified data set. */

		void dataSetChanged(data::DataSet * ds);

		/** Called when an existing data set is removed. */

		void dataSetRemoved(data::DataSet * ds);

	protected slots:
	
		/** Close the dialog. */

		void close();

		/** Sets the clicked item of the data list */
		void setClickedItem(QTreeWidgetItem*,int index);


	private:
	
		/** Update the contents of the dialog. */
		void update();

		/** Pointer to the data manager of the tool. */

		data::Manager * manager;

		/** The tree widget containing the contents of the dialog. */

		QTreeWidget * treeWidget;

		/** Button for closing the dialog. */

		QPushButton * closeButton;

		 

		/** List of all available data sets. */

		QList<data::DataSet *> dataSets;

		/** Add a new data set to the widget. 
			@param ds	New data set. */

		void populateTreeWidget(data::DataSet * ds);

		/** Create a tree item for a new data set. 
			@param ds	New data set. */

		QTreeWidgetItem * createWidgetItem(data::DataSet * ds);

		/** Add a subitem to a parent tree widget item. Commonly, the parent
			item is the main entry for a data set, while the subitem contains
			one of the properties of this data set (e.g., dimensions, etcetera). 
			@param parentItem	Parent tree widget item (main data set entry).
			@param itemText		Text for the subitem. */

		void addSubItem(QTreeWidgetItem * parentItem, QString itemText);

}; // class DataDialog


} // namespace gui


} // namespace bmia


#endif // bmia_DataDialog_h
