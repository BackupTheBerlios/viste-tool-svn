/*
 * Tool.h
 *
 * 2009-11-26	Tim Peeters
 * - First version
 */

/**
 * This header file aims to include the commonly used header
 * files of the different parts of this project. Its aim is to
 * have a single header file to include (i.e. for plugins),
 * such that all functionality becomes accessible immediately.
 *
 * Of course, if you want to optimize the compilation speed for
 * your plugin, do not include Pixie.h but only those header files
 * that you really need.
 */

#include "data/Manager.h"
#include "data/DataSet.h"
#include "data/Attributes.h"
#include "core/SimpleCoreInterface.h"
#include "core/UserOutput.h"
#include "plugin/Manager.h"
#include "plugin/Plugin.h"
#include "plugin/PluginInterfaces.h"
#include "gui/MainWindow.h"

// #include "vtkDataSet.h"
#include <vtkImageData.h>
// #include some Qt headers?
