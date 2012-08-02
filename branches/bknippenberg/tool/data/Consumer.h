/*
 * Consumer.h
 *
 * 2010-10-16	Tim Peeters
 * - First version
 */

#ifndef bmia_data_Consumer_h
#define bmia_data_Consumer_h

namespace bmia {
namespace data {

class DataSet;

/**
 * All classes that use data supplied by the data manager
 * should implement this interface class, so that the data
 * manager kan keep them up-to-date when data is
 * added/removed/changed.
 */
class Consumer {
public:
    /**
     * Here we define a virtual destructor to ensure proper destruction
     * of objects if they are to be deleted by calling the destructor
     * of this base class.
     */
    virtual ~Consumer() {};

    /**
     * This function is called when a new data set becomes available.
     *
     * @param ds The new data set that was added.
     */
    virtual void dataSetAdded(DataSet* ds) = 0;

    /**
     * This function is called when an already available data set was changed.
     *
     * @param ds The data set that has been updated.
     */
    virtual void dataSetChanged(DataSet* ds) = 0;

    /**
     * This function is called when a data set that was available has been removed.
     *
     * @param ds The data set that was removed from the pool of data sets.
     */
    virtual void dataSetRemoved(DataSet* ds) = 0;

}; // class Consumer
} // namespace data
} // namespace bmia

#endif // bmia_data_Consumer_h
