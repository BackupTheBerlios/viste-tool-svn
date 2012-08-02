/*
 * Plugin.cxx
 *
 * 2009-11-10	Tim Peeters
 * - First version.
 *
 * 2011-01-28	Evert van Aart
 * - Added additional comments.
 *
 */


/** Includes */

#include "Plugin.h"


namespace bmia {

namespace plugin {


//-----------------------------[ Constructor ]-----------------------------\\

Plugin::Plugin(QString name)
{
    this->name = name;
    this->coreInstance = NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

Plugin::~Plugin()
{

}


//-------------------------------[ getName ]-------------------------------\\

QString Plugin::getName()
{
    return this->name;
}


//---------------------------[ setCoreInstance ]---------------------------\\

void Plugin::setCoreInstance(SimpleCoreInterface * inst)
{
    this->coreInstance = inst;
}


//---------------------------------[ core ]--------------------------------\\

SimpleCoreInterface * Plugin::core()
{
    Q_ASSERT(this->coreInstance);
    return this->coreInstance;
}


} // namespace plugin


} // namespace bmia
