/*
 * DTIToolSettings.h
 *
 * 2011-07-18	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_DTIToolSettings_h
#define bmia_DTIToolSettings_h


/** Includes - Qt */

#include <QColor>


namespace bmia {


/** Simple class used to store the general settings of the DTITool. See the "Core"
	and "MainWindow" classes to see how these settings are applied. */

class DTIToolSettings
{
	public:

		/** Constructor */

		DTIToolSettings();

		/** Destructor */

		~DTIToolSettings();

		/** Initializes settings to their default values. */

		void setDefaultSettings();

		/** Position of the GUI, used for the GUI shortcuts. */

		enum GUIPosition
		{
			GUIP_Top = 0,			/**< Top field. */
			GUIP_TopExclusive,		/**< Top field (clear bottom field. */
			GUIP_Bottom				/**< Bottom field. */
		};

		/** Structure containing information a GUI shortcut. */

		struct GUIShortcutInfo
		{
			QString plugin;			/**< Plugin name. "None" if shortcut is disabled. */
			GUIPosition position;	/**< Position of the GUI (top or bottom field). */
		};

		int windowHeight;			/**< Window height in pixels. */
		int windowWidth;			/**< Window width in pixels. */
		bool maximizeWindow;		/**< Window is maximized if true. */

		QColor backgroundColor;		/**< Color of the background. */
		bool gradientBackground;	/**< Apply a gradient to the background. */

		/** Array of up to ten shortcuts for the GUI. */

		GUIShortcutInfo guiShortcuts[10];

}; // class DTIToolSettings


} // namespace bmia


#endif // bmia_DTIToolSettings_h
