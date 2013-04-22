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

#ifndef bmia_FibersToVoxels_FibersToVoxelsPlugin_h
#define bmia_FibersToVoxels_FibersToVoxelsPlugin_h

// Includes DTI tool
#include <DTITool.h>

// Includes QT
#include <QtGui>

// Includes VTK
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkLookupTable.h>

namespace bmia
{
	class FibersToVoxelsPlugin :		public plugin::Plugin,
												public data::Consumer,
												public plugin::GUI
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::data::Consumer )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		/** Constructor and destructor */
		FibersToVoxelsPlugin();
		virtual ~FibersToVoxelsPlugin();

		/** Returns plugin's QT widget */
		QWidget * getGUI();

		/** Handle dataset events */
		void dataSetAdded  ( data::DataSet * dataset );
		void dataSetRemoved( data::DataSet * dataset );
		void dataSetChanged( data::DataSet * dataset );

	private slots:

		/** Compute voxels */
		void compute();

		/** Load bootstrap fibers from file */
		void load();

		/** Load DTI volume from file. We need the dimensions and
			voxel spacing of the volume the fibers were originally
			generated from */
		void loadVolume();

        /** Load fiber scores from file. If this done, voxel values in the
            new volume will represent scores instead of normal
            visitation counts */
        void loadScores();

        /** Insert scores as point scalar data into fibers */
        void insertScores();

        /** Inserts and updates LUT for fibers */
        void updateLUT();

		/** Save voxelized fibers to file */
		void save();

        /** Save scored fibers */
        void saveFibers();

        /** Undo NIFTI transform on the fibers */
        void undoTransform();
        void transform();

        /** Sets new score */
        void scoreChanged( int );

        /** Sets new opacity */
        void opacityChanged( int );

        /** Compute fiber confidence table */
        void computeScoresFromDistances();

        /** Compute scores for arbitrary set of fibers based on voxel intersection counts */
        void computeScores();

        /** Inverts scores by storing first score then cell ID */
        void invertScores();

        /** Converts seed points from raw bytes to Camino text format */
        void seedPointsToText();

	private:

        float getFloat32(std::ifstream & f)
        {
            char bytes[sizeof(float)];
            f.read(bytes, sizeof(bytes));

            float value;
            char * bytesSwitched = (char *) & value;
            bytesSwitched[0] = bytes[3];
            bytesSwitched[1] = bytes[2];
            bytesSwitched[2] = bytes[1];
            bytesSwitched[3] = bytes[0];

            return value;
        }

        class Point
        {
        public:
            Point() { x = 0; y = 0; z = 0; }
            Point( float xx, float yy, float zz )
            { x = xx; y = yy; z = zz; }
            float x, y, z;
        };

        QList<double> computeNormalizedSumOfPairwiseDistances
            (QList<QList<Point> > & fibers);

        double computeMeanOfClosestPointDistance
            (QList<Point> & fiberA, QList<Point> & fiberB);

        double computeEndPointDistance
            (QList<Point> & fiberA, QList<Point> & fiberB);

        double computeClosestPointDistance
            (const Point & p, QList<Point> & fiber);

        double computeDistance
            (const Point & pA, const Point & pB);

        void writeScoresToFile(int index, QList<double> & scores, std::ofstream & f);

        QPushButton	* _button;					// The button for computing voxels
		QPushButton * _buttonSave;				// The button for saving voxelized fibers to file
		QPushButton * _buttonLoad;				// The button for loading fibers from file
		QPushButton * _buttonLoadVolume;		// The button for loading fibers from file
        QPushButton * _buttonUndoTransform;     // The button for inverting the NIFTI transform in the fibers
        QPushButton * _buttonInsertScores;      // The button for inserting scores into fiber's point scalar data
        QPushButton * _buttonSaveFibers;        // The button for saving scored fibers
        QPushButton * _buttonInsertLUT;
        QPushButton * _buttonInvertScores;
        QPushButton * _buttonComputeScores;
        QPushButton * _buttonComputeScoresFromDistances;
        QPushButton * _buttonSeedPointsToText;
		QComboBox	* _fiberBox;				// The combobox with fiber tract names
		QComboBox	* _datasetBox;				// The combobox with volume dataset names
        QComboBox   * _distanceMeasureBox;
        QComboBox   * _densityBox;

        QSlider     * _sliderScores;            // The slider for manipulating scores
        QSlider     * _sliderOpacity;

		QCheckBox   * _binaryBox;
		QLineEdit   * _dimEdit[3];
		QLineEdit   * _spacingEdit[3];
        QLineEdit   * _colorEdit[3];
        QLineEdit   * _nrIterationsEdit;
        QLineEdit   * _nrSeedPointsEdit;
		QBoxLayout	* _layout;					// The button layout
		QWidget		* _widget;					// The QT widget holding this plugin's GUI

        QList< double > _fiberScores;
		QList< vtkPolyData * >	_fiberList;		// The list with fiber tracts
		QList< vtkImageData * >	_datasetList;	// The list with volume datatsets

        vtkMatrix4x4 * _matrix;
        vtkLookupTable * _fiberLUT;
	};
}

#endif
