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

#ifndef bmia_ImageTypeConverter_ImageTypeConverterPlugin_h
#define bmia_ImageTypeConverter_ImageTypeConverterPlugin_h

// Includes DTI tool
#include <DTITool.h>

// Includes QT
#include <QtGui>

// Includes VTK
#include <vtkTransform.h>
#include <vtkImageData.h>

namespace bmia
{
	class ImageTypeConverterPlugin :	public plugin::Plugin,
										public data::Consumer,
										public plugin::GUI
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::data::Consumer )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		/** Constructor and destructor */
		ImageTypeConverterPlugin();
		virtual ~ImageTypeConverterPlugin();

		/** Returns plugin's QT widget */
		QWidget * getGUI();

		/** Handle dataset events */
		void dataSetAdded  ( data::DataSet * dataset );
		void dataSetRemoved( data::DataSet * dataset );
		void dataSetChanged( data::DataSet * dataset );

	private slots:

		/** Executes conversion */
		void convert();

        /** Pads dataset to nearest power of two */
        void pad();

		/** Saves dataset to .VOL format */
		void save();

	private:

        /** Returns next power of two */
        int NextPowerOfTwo( int n );

		QComboBox   * _datasetBox;			// Combobox containing names of datasets
		QComboBox   * _typeBox;				// Combobox containing data types
		QPushButton * _button;				// Button for starting coordinate transformation
        QPushButton * _buttonSave;          // Button for saving dataset to .VOL
        QPushButton * _buttonPad;           // Button for padding dataset to nearest power of two
		QVBoxLayout * _layout;				// The layout of the widget
		QWidget     * _widget;				// Widget containing UI

		QString _kind;
		QList< vtkImageData * > _datasets;
	};
}

#endif
