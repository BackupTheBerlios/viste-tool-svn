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
 * VtiReaderPlugin.h
 *
 * 2010-11-09	Tim Peeters
 * - First version
 *
 * 2011-01-27	Evert van Aart
 * - Added support for loading transformation matrices
 *
 */

#ifndef bmia_VtiReader_VtiReaderPlugin_h
#define bmia_VtiReader_VtiReaderPlugin_h

#include "DTITool.h"

namespace bmia {

/**
 * A plugin for reading VTK imagedata files.
 */
class VtiReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

public:
    VtiReaderPlugin();
    ~VtiReaderPlugin();

    /**
     * Returns the list of file extensions supported by this reader plugin.
     * This function is required by the data::Reader plugin interface.
     */
    QStringList getSupportedFileExtensions();

	/** Returns a list containing short descriptions of the supported file
		types. The number of descriptions and their order should match those
		of the list returned by "getSupportedFileExtensions". */

	QStringList getSupportedFileDescriptions();

	/**
     * Load geometry data from the given file and make it available
     * to the data manager.
     * This function is required by the data::Reader plugin interface.
     */
    void loadDataFromFile(QString filename);

protected:
private:

}; // class VtiReaderPlugin
} // namespace bmia
#endif // bmia_VtiReader_VtiReaderPlugin_h
