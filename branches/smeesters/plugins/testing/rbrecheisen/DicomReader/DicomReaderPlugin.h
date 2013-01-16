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

#ifndef bmia_DicomReader_DicomReaderPlugin_h
#define bmia_DicomReader_DicomReaderPlugin_h

// Includes DTI tool
#include <DTITool.h>

// Includes QT
#include <QList>
#include <QString>
#include <QStringList>

// Includes GDCM
#include <vtkGDCMImageReader.h>

/** @class DicomReaderPlugin
	@brief This plugin allows reading of DICOM datasets */
namespace bmia
{
	class DicomReaderPlugin :	public plugin::Plugin,
								public plugin::GUI,
								public data::Reader
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::data::Reader )

	public:

		/** Constructor and destructor */
		DicomReaderPlugin();
		virtual ~DicomReaderPlugin();

		/** Returns plugin's QT widget */
		QWidget * getGUI();

		/** Returns plugin's supported file extensions */
		QStringList getSupportedFileExtensions();

		/** Returns short description for each supported file type */
		QStringList getSupportedFileDescriptions();

		/** Loads the DICOM data */
		void loadDataFromFile( QString fileName );

	private:

		QWidget * _widget;
		QList< vtkGDCMImageReader * > _readers;
	};
}

#endif
