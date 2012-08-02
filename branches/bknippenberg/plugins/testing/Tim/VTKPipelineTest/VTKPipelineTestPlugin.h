/*
 * VTKPipelineTestPlugin.h
 *
 * 2010-05-30	Tim Peeters
 * - First version
 */

#ifndef bmia_DTIMeasures_VTKPipelineTestPlugin_h
#define bmia_DTIMeasures_VTKPipelineTestPlugin_h

#include "DTITool.h"

namespace bmia {
/**
 * A plug-in that computes DTI anisotropy measures from a tensor volume dataset.
 */
class VTKPipelineTestPlugin : public plugin::Plugin, public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)

public:
    VTKPipelineTestPlugin();
    ~VTKPipelineTestPlugin();

    /**
     * Consumer interface functions.
     */
    virtual void dataSetAdded(data::DataSet* ds);
    virtual void dataSetChanged(data::DataSet* ds);
    virtual void dataSetRemoved(data::DataSet* ds);

protected:

private:


}; // class VTKPipelineTestPlugin
} // namespace bmia
#endif // bmia_DTIMeasures_VTKPipelineTestPlugin_h
