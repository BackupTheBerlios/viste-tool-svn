/**
 * DataInterfaces.h
 *
 * 2009-11-04	Tim Peeters
 * - First version.
 *
 */


/** Includes */

#include "data/Reader.h"
#include "data/Consumer.h"


#ifndef bmia_plugin_DataInterfaces_h
#define bmia_plugin_DataInterfaces_h


Q_DECLARE_INTERFACE(bmia::data::Reader,   "bmia.data.Reader")
Q_DECLARE_INTERFACE(bmia::data::Consumer, "bmia.data.Consumer")


#endif // bmia_plugin_DataInterfaces_h
