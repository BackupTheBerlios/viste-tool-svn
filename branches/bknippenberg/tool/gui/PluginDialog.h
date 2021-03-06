/*
 * PluginDialog.h
 *
 * 2010-02-11	Tim Peeters
 * - First version.
 *
 * 2011-01-28	Evert van Aart
 * - Added comments, minor code refactoring.
 *
 * 2011-07-21	Evert van Aart
 * - Added destructor.
 *
 */


#ifndef bmia_gui_PluginDialog_h
#define bmia_gui_PluginDialog_h


/** Includes - Custom Files */

#include "plugin/Manager.h"
#include "plugin/Plugin.h"
#include "plugin/PluginInterfaces.h"
#include "core/Core.h"
#include "core/UserOutput.h"

/** Includes - Qt */

#include <QDialog>
#include <QDir>
#include <QTextStream>
#include <QPushButton>
#include <QFileDialog>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QtDebug>

/** Includes - C++ */

#include <assert.h>


namespace bmia {


namespace gui {


/** This class implements the plugin dialog window that lists all available plugins. From
	this window, the user can load and unload available plugins by clicking the checkbox next
	to the plugin name. The window also contains a button for loading new plugins, and a
	button for cleaning up the list by removing all non-active plugins. Expanding a plugin
	gives additional information, such as its version and the plugin type (e.g., "Consumer",
	"Visualization", etcetera).
*/

class PluginDialog : public QDialog
{
	Q_OBJECT

	public:
    
		/** Construct a new plugin dialog with the given plugin manager. The plugin manager 
			may not be NULL and must stay active while this plugin dialog exists.
			@param pManager		Plugin manager.
			@param parent		Parent widget. */
     
		PluginDialog(plugin::Manager * pManager, QWidget * parent = NULL);

		/** Destructor */

		~PluginDialog();

		/** (Re)build the list of plugins without removing the inactive ones. */

		void rebuildList();

		/** Set a new plugin directory. 
			@param dir			Full path to the plugin directory. */

		void setPluginDir(QString dir)
		{
			pluginDir = dir;
		}

	public slots:

		/** Remove all unloaded plugins from the plugin manager and refresh the list of plugins. */
    
		void clean();

	protected slots:

		/** Let the user load a new plugin from a file. */
    
		void add();

		/** Called when the user checks or unchecks a plugin. 
			@param item			The list item that was changed. 
			@param column		Column index, not used. */

		void onItemChanged(QTreeWidgetItem * item, int column);

	private:
    
		/** Add information and Qt widgets for one plugin.
			@param i			Plugin index. */

		void populateTreeWidget(int i);

		/** Plugin manager. */

		plugin::Manager * manager;

		/** The tree widget displayed in the dialog window. */

		QTreeWidget * treeWidget;

		/** Button for cleaning up the list. */
    
		QPushButton * cleanButton;

		/** Button for adding a plugin. */
    
		QPushButton * addButton;

		/** Directory containing the plugins. The first time a loaded plugin is added to
			the plugin dialog, its directory is copied to this string; when a user presses
			the "Add" button, this directory is used as the default search path. */

		QString pluginDir;

}; // class PluginDialog


} // namespace gui


} // namespace bmia


#endif // bmia_gui_PluginDialog_h
