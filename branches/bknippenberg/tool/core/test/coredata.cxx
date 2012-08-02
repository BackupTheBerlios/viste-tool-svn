/*
 * coredata.cxx
 *
 * 2009-11-09	Tim Peeters
 * - First version
 */
#include "core/Core.h"
#include "data/Manager.h"
#include "data/DataSet.h"
using namespace bmia;

/**
 * Test core and data manager.
 */
int main(int argc, char ** argv)
{
    data::DataSet* d1 = new data::DataSet("First data set", "boring type");
    data::DataSet* d2 = new data::DataSet("Second data set", "boring type");

    Core* core = new Core();
    core->data()->addDataSet(d1);
    core->data()->addDataSet(d2);
    core->data()->printAllDataSets();
    core->data()->removeDataSet(d1);
    core->data()->printAllDataSets();

    QList<data::DataSet*> sets = core->data()->listDataSets("boring type");
    return (sets.size() != 1); // pass the test if there is 1 data set of type "boring type"
}
