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


/** All plugin classes that use data supplied by the data manager should implement 
	this interface class, so that the data manager can keep them up-to-date when 
	data is added/removed/changed.
*/


class Consumer 
{
	public:

		/**	Destructor. Here we define a virtual destructor to ensure proper 
			destruction of objects if they are to be deleted by calling the 
			destructor of this base class. */
    
		virtual ~Consumer() {};

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

	    virtual void dataSetAdded(DataSet * ds) = 0;

		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */
	
		virtual void dataSetChanged(DataSet * ds) = 0;

		/** The data manager calls this function whenever an existing
			data set is removed. */
	
		virtual void dataSetRemoved(DataSet * ds) = 0;

}; // class Consumer


} // namespace data


} // namespace bmia


#endif // bmia_data_Consumer_h
