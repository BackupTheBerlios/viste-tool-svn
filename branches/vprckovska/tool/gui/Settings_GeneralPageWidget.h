/*
 * Settings_GeneralPageWidget.h
 *
 * 2011-07-13	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_Settings_GeneralPageWidget_h
#define bmia_Settings_GeneralPageWidget_h


/** Includes - Custom Files */

#include "Settings_GenericPageWidget.h"

/** Includes - Qt */

#include <QGroupBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QLabel>
#include <QSpacerItem>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QColorDialog>
#include <QPushButton>
#include <QCheckBox>


namespace bmia {


/** This class represents a settings page containing general settings, like
	the window size. It inherits its core functionality from "Settings_GenericPageWidget".
*/

class Settings_GeneralPageWidget : public Settings_GenericPageWidget
{
	Q_OBJECT

	public:

		/** Constructor */

		Settings_GeneralPageWidget();

		/** Destructor */

		~Settings_GeneralPageWidget();

		/** Return the name of this page. */

		virtual QString getPageName();

		/** Copy current settings to GUI controls. 
			@param settings	Input DTITool settings. */

		virtual void initializeControls(DTIToolSettings * settings);

		/** Copy GUI control values back to the settings; return true if the settings
			were modified, and false otherwise. 
			@param settings	Output DTITool settings. */

		virtual bool storeSettings(DTIToolSettings * settings);

	protected:

		QRadioButton * pMaximizeWindowRadio;	/**< Radio button for maximizing the window. */
		QRadioButton * pCustomSizeRadio;		/**< Radio button for setting a custom window size. */
		QSpinBox * pWindowWidthSpin;			/**< Spinner for the window width. */
		QSpinBox * pWindowHeightSpin;			/**< Spinner for the window height. */
		QFrame * pBGColorFrame;					/**< Frame displaying the current background color. */
		QCheckBox * pBGGradientCheck;			/**< Check box for gradient background colors. */

	protected slots:

		/** Use a color dialog to pick a new background color. */

		void pickColor();

}; // class Settings_GeneralPageWidget



} // namespace bmia


#endif // bmia_Settings_GeneralPageWidget_h