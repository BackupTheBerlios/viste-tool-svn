/*
 * SimpleCoreInterface
 *
 * 2010-05-04	Tim Peeters	
 * - First version
 *
 * 2011-04-06	Evert van Aart
 * - Added "getDataDirectory".
 *
 * 2011-04-14	Evert van Aart
 * - Added "disableRendering" and "enableRendering".
 *
 */

#ifndef bmia_SimpleCoreInterface_h
#define bmia_SimpleCoreInterface_h

#include <QDir>

namespace bmia {

class UserOutput;
namespace data { class Manager; }

/**
 * Interface for the plugins to access core.
 */
class SimpleCoreInterface
{
public:
    /**
     * Return the data manager.
     */
    virtual data::Manager* data() = 0;

    /**
     * Return the UserOutput object.
     */
    virtual UserOutput* out() = 0;

    /**
     * Call this function to re-draw the scene after making
     * updates in the settings or pipeline.
     */
    virtual void render() = 0;

	virtual QDir getDataDirectory() = 0;

	/** Turn off rendering. If rendering is turned off, any call to the "render"
		function will immediately return. Extreme care should be taken when using
		this function: Every call to "disableRendering" should be followed by a
		call to "enableRendering" eventually! The purpose of this function is to
		avoid redundant re-renders when changing a large amount of data. For example,
		switching to a new image in the Planes plugin updates the three slice actors
		and the three plane seed point sets; each of these updates can trigger a
		render call in one (or more!) other plugins. Therefore, we turn off rendering
		before we switch to the new image, and re-enable when we're done (followed
		by a single render call. */

	virtual void disableRendering() = 0;

	/** Turn on rendering. See notes for "disableRendering". */

	virtual void enableRendering() = 0;

}; // class SimpleCoreInterface
} // namespace bmia
#endif // bmia_SimpleCoreInterface_h
