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
 * PDBReaderPlugin.h
 *
 * 2013-05-16	Stephan Meesters
 * - First version
 *
 */

/** Includes */

#include "PDBReaderPlugin.h"



namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

PDBReaderPlugin::PDBReaderPlugin() : plugin::Plugin("PDB Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

PDBReaderPlugin::~PDBReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList PDBReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;
    list.push_back("pdb");
    return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList PDBReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("PDB files");
	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void PDBReaderPlugin::loadDataFromFile(QString filename)
{
	// Print status message to the log
	this->core()->out()->logMessage("Trying to load data from file " + filename);

	// Create the Qt file handler
	ifstream pdbFile (filename.toUtf8().constData(), ios::in | ios::binary );

	// Try to open the input file
	if (pdbFile.fail())
	{
		this->core()->out()->logMessage("Could not open file " + filename + "!");
		return;
	}

    // temp variables
    int ii; double d; char c; short s;

    // load header size
    unsigned int headersize;
    pdbFile.read(reinterpret_cast<char*>(&headersize), sizeof(unsigned int));
    //std::cout << "headersize: " << headersize << std::endl;

    // load transformation matrix
    QMatrix4x4 matrix;
    //double matrix[4][4];
    for(int i =0; i<4; i++)
    {
        for(int j = 0; j<4; j++)
        {
            pdbFile.read(reinterpret_cast<char*>(&d), sizeof(double));
            //matrix[i][j] = d;
            matrix(i,j) = d;
            //std::cout << "matrix " << d << std::endl;
        }
    }

    // load number of stats
    int numstats;
    pdbFile.read(reinterpret_cast<char*>(&numstats), sizeof(int));
    //std::cout << "number of stats: " << numstats << std::endl;

    // load statheaders
    StatHeader* statheaders = new StatHeader[numstats];
    for(int i = 0; i<numstats; i++)
    {
        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        statheaders[i].i1 = ii;

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        statheaders[i].i2 = ii;

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        statheaders[i].i3 = ii;

        for(int j = 0; j<255; j++)
        {
            pdbFile.read(&c, sizeof(char));
            statheaders[i].c1[j] = c;
        }

        for(int j = 0; j<255; j++)
        {
            pdbFile.read(&c, sizeof(char));
            statheaders[i].c2[j] = c;
        }

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        statheaders[i].i4 = ii;

        pdbFile.read(reinterpret_cast<char*>(&s), sizeof(short)); // dummy

        //std::cout << "statheaders.i1: " << statheaders[i].i1 << std::endl;
        //std::cout << "statheaders.i2: " << statheaders[i].i2 << std::endl;
        //std::cout << "statheaders.i3: " << statheaders[i].i3 << std::endl;
        //std::cout << "statheaders.c1: " << statheaders[i].c1 << std::endl;
        //std::cout << "statheaders.c2: " << statheaders[i].c2 << std::endl;
        //std::cout << "statheaders.i4: " << statheaders[i].i4 << std::endl;
    }

    // count number of point stats
    int numPointStats = 0;
    for(int i = 0; i<numstats; i++)
    {
        if(statheaders[i].i2 == 1)
            numPointStats++;
    }
    bool usingPointStats = numPointStats > 0;
    if(!usingPointStats)
        this->core()->out()->logMessage("PDB file contains no scoring values.");
    //std::cout << "number of point stats: " << numPointStats << std::endl;

    // number of algo's
    int numAlgos;
    pdbFile.read(reinterpret_cast<char*>(&numAlgos), sizeof(int));
    //std::cout << "number of algos: " << numAlgos << std::endl;

    // load algoheaders
    AlgoHeader* algoheaders = new AlgoHeader[numAlgos];
    for(int i = 0; i<numAlgos; i++)
    {
        for(int j = 0; j<255; j++)
        {
            pdbFile.read(&c, sizeof(char));
            algoheaders[i].c1[j] = c;
        }

        for(int j = 0; j<255; j++)
        {
            pdbFile.read(&c, sizeof(char));
            algoheaders[i].c2[j] = c;
        }

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        algoheaders[i].i1 = ii;

        pdbFile.read(reinterpret_cast<char*>(&s), sizeof(short)); // dummy

        //std::cout << "algoheaders.c1: " << algoheaders[i].c1 << std::endl;
        //std::cout << "algoheaders.c2: " << algoheaders[i].c2 << std::endl;
        //std::cout << "algoheaders.i1 " << algoheaders[i].i1 << std::endl;
    }

    // version number
    int versionNumber;
    pdbFile.read(reinterpret_cast<char*>(&versionNumber), sizeof(int));
    //std::cout << "version number: " << versionNumber << std::endl;

    // skip to end of header
    pdbFile.seekg(headersize, pdbFile.beg);

    // number of pathways
    int numPathways;
    pdbFile.read(reinterpret_cast<char*>(&numPathways), sizeof(int));
    //std::cout << "number of pathways: " << numPathways << std::endl;

    // load pathways
    Pathway* pathways = new Pathway[numPathways];
    int totalNumberOfPoints = 0;
    for(int i = 0; i<numPathways; i++)
    {
        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        pathways[i].headerSize = ii;

        int pos = pdbFile.tellg();

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        pathways[i].numPoints = ii;

        totalNumberOfPoints += pathways[i].numPoints;

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        pathways[i].algoInt = ii;

        pdbFile.read(reinterpret_cast<char*>(&ii), sizeof(int));
        pathways[i].seedPointIndex = ii;

        //std::cout << "pathways.numPoints: " << pathways[i].numPoints << std::endl;
        //std::cout << "pathways.algoInt " << pathways[i].algoInt << std::endl;
        //std::cout << "pathways.seedPointIndex " << pathways[i].seedPointIndex << std::endl;

        pathways[i].pathStats = new double[numstats];
        for(int j = 0; j<numstats; j++)
        {
            pdbFile.read(reinterpret_cast<char*>(&d), sizeof(double));
            pathways[i].pathStats[j] = d;
            //std::cout << "pathStats:" << j << " = " << d << std::endl;
        }

        // skip to end of header of pathway
        pdbFile.seekg(pathways[i].headerSize + pos, pdbFile.beg);

        pathways[i].points = new double[pathways[i].numPoints*3]; // 3D points
        for(int j = 0; j<pathways[i].numPoints; j++)
        {
            for(int k =0; k<3; k++)
            {
                pdbFile.read(reinterpret_cast<char*>(&d), sizeof(double));
                pathways[i].points[j*3+k] = d;
            }
            //std::cout << "point" << pathways[i].points[j*3+0] << "," << pathways[i].points[j*3+1] << "," << pathways[i].points[j*3+2] << std::endl;
        }

        if(usingPointStats)
        {
            pathways[i].pointStats = new double[pathways[i].numPoints * numPointStats]; // scalar score per point
            int index = 0;
            for(int k =0; k<numPointStats; k++)
            {
                for(int j = 0; j<pathways[i].numPoints; j++)
                {
                    pdbFile.read(reinterpret_cast<char*>(&d), sizeof(double));
                    pathways[i].pointStats[k*pathways[i].numPoints + j] = d;
                    //std::cout << "pointstat type:" << k << " -- " << d << std::endl;
                }
            }
        }
    }

    // close .pdb file
    pdbFile.close();

    // notify load complete
    this->core()->out()->logMessage("PDB processed: " + filename);

    // create struct to hold fibers
    Fibers fibers;
    fibers.filename = filename;
    fibers.matrix = matrix;
    fibers.numstats = numstats;
    fibers.statheaders = statheaders;
    fibers.numPointStats = numPointStats;
    fibers.numAlgos = numAlgos;
    fibers.algoheaders = algoheaders;
    fibers.versionNumber = versionNumber;
    fibers.numPathways = numPathways;
    fibers.pathways = pathways;

    //
    //  Transformation
    //
    for(int i = 0; i<numPathways; i++)
    {
        for(int j = 0; j<pathways[i].numPoints; j++)
        {
            // translation
            pathways[i].points[j*3+0] += 80;
            pathways[i].points[j*3+1] += 120;
            pathways[i].points[j*3+2] += 60;

            // scaling
            pathways[i].points[j*3+0] *= 0.5;
            pathways[i].points[j*3+1] *= 0.5;
            pathways[i].points[j*3+2] *= 0.5;
        }
    }


    vtkMatrix4x4 * mat = vtkMatrix4x4::New();
    mat->Identity();

    vtkTransform* transform = vtkTransform::New();
    transform->SetMatrix(mat);
    transform->Translate(-80,-120,-60);
    transform->Scale(2,2,2);
    mat = transform->GetMatrix();

    //
    //  VTK conversion
    //

    // Create polydata
    vtkPolyData* output = vtkPolyData::New();

	// Create a point set for the output
	vtkPoints * outputPoints = vtkPoints::New();
	output->SetPoints(outputPoints);
	outputPoints->Delete();

    // Cell array holding the pathways.
    // Each pathway is a single cell (line).
    // Each cell holds the id values to points from the vtkPoints list
    vtkCellArray * outputLines = vtkCellArray::New();
    output->SetLines(outputLines);
    outputLines->Delete();

    // Array holding the ConTrack scoring values
    vtkDoubleArray* scoring;
    if(usingPointStats)
    {
        scoring = vtkDoubleArray::New();
        scoring->SetName("Contrack_Score");
        scoring->SetNumberOfTuples(totalNumberOfPoints);
    }

    // Loop over pathways
    int counter = 0;
    for(int i = 0; i < numPathways; i++)
    {
        int numberOfFiberPoints = pathways[i].numPoints;

        // Create a cell representing a fiber
        outputLines->InsertNextCell(numberOfFiberPoints);

        // Loop over points in the pathway
        for(int j = 0; j<numberOfFiberPoints; j++)
        {
            outputPoints->InsertNextPoint(pathways[i].points[j*3],pathways[i].points[j*3+1],pathways[i].points[j*3+2]);
            outputLines->InsertCellPoint(counter + j);
            if(usingPointStats)
            {
                scoring->SetTuple1(counter + j, pathways[i].pointStats[1*numberOfFiberPoints + j]);
            }
        }

        counter += numberOfFiberPoints;
    }

    // Set active scalars to ConTrack score
    if(usingPointStats)
    {
        output->GetPointData()->AddArray(scoring);
        output->GetPointData()->SetActiveScalars("Contrack_Score");
    }

    //
    //  Save to dataset
    //

    // Short name of the data set
	QString shortName = filename;

	// Find the last slash
	int lastSlash = filename.lastIndexOf("/");

	// If the filename does not contain a slash, try to find a backslash
	if (lastSlash == -1)
	{
		lastSlash = filename.lastIndexOf("\\");
	}

	// Throw away everything up to and including the last slash
	if (lastSlash != -1)
	{
		shortName = shortName.right(shortName.length() - lastSlash - 1);
	}

	// Find the last dot in the remainder of the filename
	int lastPoint = shortName.lastIndexOf(".");

	// Throw away everything after and including the last dot
	if (lastPoint != -1)
	{
		shortName = shortName.left(lastPoint);
	}

	// Create a new data set for the transfer function
	data::DataSet * ds = new data::DataSet(shortName, "fibers", output);

	// Fibers should be visible, and the visualization pipeline should be updated
	ds->getAttributes()->addAttribute("isVisible", 1.0);
	ds->getAttributes()->addAttribute("updatePipeline", 1.0);

	// Copy the transformation matrix to the output
	ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(mat));

    // Add the data set to the manager
	this->core()->data()->addDataSet(ds);
}

} // namespace bmia


Q_EXPORT_PLUGIN2(libPDBReaderPlugin, bmia::PDBReaderPlugin)
