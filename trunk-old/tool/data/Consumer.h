/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
