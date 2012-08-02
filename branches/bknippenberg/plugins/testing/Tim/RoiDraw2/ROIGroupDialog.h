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