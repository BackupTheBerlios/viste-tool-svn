/*
 * AdvancedPlugin.h
 *
 * 2010-05-04	Tim Peeters
 * - First version
 *
 */


/** Includes */

#include "AdvancedPlugin.h"
#include "core/Core.h"


namespace bmia {


namespace plugin {


//-----------------------------[ Constructor ]-----------------------------\\

AdvancedPlugin::AdvancedPlugin(QString name) : Plugin(name)
{
	this->fullCoreInstance = NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

AdvancedPlugin::~AdvancedPlugin()
{

}


//-------------------------[ setFullCoreInstance ]-------------------------\\

void AdvancedPlugin::setFullCoreInstance(Core * inst)
{
	this->fullCoreInstance = inst;
}


//-------------------------------[ fullCore ]------------------------------\\

Core * AdvancedPlugin::fullCore()
{
	Q_ASSERT(this->fullCoreInstance);
	return this->fullCoreInstance;
}


} // namespace plugin


} // namespace bmia
