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
 * Manager.cxx
 *
 * 2009-10-27	Tim Peeters
 * - First version
 *
 * 2009-11-04	Tim Peeters
 * - Add printAllDataSets() function.
 *
 * 2010-03-11	Tim Peeters
 * - Do not add a data set when a data set with the same
 *   name was added before. This avoids consumers (e.g. DTIMeasures)
 *   that are reloaded to add the same data again.
 *
 * 2011-01-03	Evert van Aart
 * - Added a function that combines the supported extensions of the readers
 *   with the file descriptions, into a format that can be used as the filter
 *   string of a file dialog.
 *
 */

#include "Manager.h"
#include "DataSet.h"
#include "Consumer.h"
#include "Reader.h"
#include "Writer.h"

#include <QFile>
#include <QTextStream>
#include <QDebug>

#include <assert.h>
#include <iostream>
using namespace std;

namespace bmia {
namespace data {

Manager::Manager()
{
    // nothing to do.
    Q_ASSERT(this->dataSets.size() == 0);
    Q_ASSERT(this->consumers.size() == 0);
    Q_ASSERT(this->readers.size() == 0);
	 Q_ASSERT(this->writers.size() == 0);
}

Manager::~Manager()
{
    // dataSets, consumers and readers will be cleared.
    // TODO: this changed. REMOVE data sets!
}

bool Manager::addDataSet(DataSet* ds)
{
    Q_ASSERT(ds);

    // check whether the data set was not in the list yet.
    // if the data set was already in the list, do nothing
    if (this->dataSets.contains(ds)) return false;

    /* (name, kind) no longer needs to be unique. Use DataSet pointers to distinguish data sets!

    // (name, kind) must be unique, so do not add the new data set if
    // there is already a data set with the same name.
    QString name = ds->getName();
    QString kind = ds->getKind();
    for (int i = 0; i < this->dataSets.size(); i++)
	{
	if ((this->dataSets[i]->getKind() == kind) && (this->dataSets[i]->getName() == name))
	    {
	    qDebug()<<"Data set was not added because there is already data with name"<<name<<"and kind"<<kind;
	    return false;
	    } // if
	} // for i
    */

    // add the data set to the list of data sets:
    this->dataSets.push_back(ds);

    // notify all consumers of the added dat set:
    int n = this->consumers.size();
    for (int i = 0; i < n; i++)
        {
        this->consumers.at(i)->dataSetAdded(ds);
        } // for i

    return true;
}

void Manager::removeDataSet(DataSet* ds)
{
   Q_ASSERT(ds);
    int i; // counter

    // check whether the data set is currently in the list
    i = this->dataSets.indexOf(ds);
    if (i == -1) return;

    // i is the index of ds in the list of data sets
    this->dataSets.removeAt(i);

    // notify the consumers:
    int n = this->consumers.size();
    for (i = 0; i < n; i++)
	{
	this->consumers.at(i)->dataSetRemoved(ds);
	} // for j

    delete ds;
}

void Manager::dataSetChanged(DataSet* ds)
{
    // check whether the data set is in the list
    if (!this->dataSets.contains(ds)) return;

    for (int i = 0; i < this->consumers.size(); i++)
	{
	this->consumers.at(i)->dataSetChanged(ds);
	} // for i
}

QList<DataSet*> Manager::listDataSets(QString kind)
{
    QList<DataSet*> dss; // vector to return. start empty.
    int n = this->dataSets.size();

    // add all the data sets ds for which (ds->getKind() == kind)
    // to the output list dss.
    DataSet* ds;
    for (int i = 0; i < n; i++)
	{
	ds = this->dataSets.at(i);
	Q_ASSERT(ds);
	if (ds->getKind() == kind) dss.push_back(ds);
	} // for
    return dss;
}

QList<DataSet*> Manager::listAllDataSets()
{
    return this->dataSets;
}

DataSet* Manager::getDataSet(QString name, QString kind)
{
    int i; int j = -1;
    for (i = 0; i < this->dataSets.size(); i++)
	{
	if ((this->dataSets[i]->getKind() == kind) && (this->dataSets[i]->getName() == name)) j = i;
	} // for i
    if (j == -1) return NULL;
    return this->dataSets[j];
}

void Manager::printAllDataSets()
{
    int n = this->dataSets.size();
    QTextStream out(stdout);
    out<<"Number of data sets = "<<n<<endl;

    out<<"All data sets in the data manager: "<<endl;
    DataSet* ds;
    for (int i = 0; i < n; i++)
	{
	ds = this->dataSets.at(i);
	Q_ASSERT(ds);
	out<<"- "<<i<<" ("<<ds<<"): "<<endl;
	out<<"  "<<ds->getName()<<" ("<<ds->getKind()<<")"<<endl;
	} // for i
}

void Manager::printSupportedFileExtensions()
{
    QTextStream out(stdout);
    QStringList extensions = this->getSupportedFileExtensions();
    out<<"Supported file extensions:";
    for (int i = 0; i < extensions.size(); i++)
	{
	out<<" "<<extensions.at(i);
	} // for i
    out<<endl;
}

void Manager::addConsumer(Consumer* cons)
{
    Q_ASSERT(cons);
    if (!this->consumers.contains(cons)) this->consumers.push_back(cons);
    // now notify the new consumer of all data sets already available
    for (int i=0; i < this->dataSets.size(); i++)
	{
	cons->dataSetAdded(this->dataSets[i]);
	} // for i
}

void Manager::removeConsumer(Consumer* cons)
{
    Q_ASSERT(cons);
    int i = this->consumers.indexOf(cons);
    if (i == -1) return;
    this->consumers.removeAt(i);
}

void Manager::addReader(Reader* reader)
{
    if (!this->readers.contains(reader)) this->readers.push_back(reader);
}

void Manager::addWriter(Writer *writer)
{
    if (!this->writers.contains(writer)) this->writers.push_back(writer);
}

void Manager::removeReader(Reader* reader)
{
    int i = this->readers.indexOf(reader);
    if (i == -1) return;
    this->readers.removeAt(i);
}
void Manager::removeWriter(Writer* writer)
{
    int i = this->writers.indexOf(writer);
    if (i == -1) return;
    this->writers.removeAt(i);
}

QStringList Manager::getSupportedFileExtensions()
{
    QStringList ext1;	// extensions per reader
    QStringList extAll;	// concatenation of all ext1's

    Reader* r;
    for (int i=0; i < this->readers.size(); i++)
	{
	r = this->readers.at(i);
	ext1 = r->getSupportedFileExtensions();
	for (int j = 0; j < ext1.size(); j++)
	    {
	    extAll.push_back(ext1.at(j));
	    } // for j
	} // for i
    return extAll;
}


QStringList Manager::getSupportedFileExtensionsWithDescriptions()
{
	// List of extensions for one reader
	QStringList extensions;

	// List of file descriptions for one reader
	QStringList descriptions;

	// Combined output
	QStringList output;

	// One string containing all supported file types (default selection)
	QString allFiles = "All (";

	Reader * r;

	// Loop through all readers
	for (int i = 0; i < this->readers.size(); ++i)
	{
		// Get the current reader
		r = this->readers.at(i);

		// Get the supported extensions and descriptions
		extensions   = r->getSupportedFileExtensions();
		descriptions = r->getSupportedFileDescriptions();

		// Extension list and description list should have the same size. If you're
		// coding a new plugin and you triggered this assert, you have probably hard-
		// coded extension/description lists of different sizes.

		assert(extensions.size() == descriptions.size());
		
		// Loop through all supported file types for this reader
		for (int j = 0; j < extensions.size(); ++j)
		{
			// Create the combined output
			output.push_back(descriptions.at(j) + " (*." + extensions.at(j) + ")");

			// Append extension to the filter for all supported file types
			allFiles.append("*." + extensions.at(j) + " ");
		}
	}

	// Remove the last character (space), and add the closing parenthesis
	allFiles.resize(allFiles.length() - 1);
	allFiles.append(")");

	// Sort the list by alphabetic order
	output.sort();

	// Add the "allFiles" string to the front (default option)
	output.push_front(allFiles);

	return output;
}
QStringList Manager::getSupportedFileWriteExtensions()
{
    QStringList ext1;	// extensions per reader
    QStringList extAll;	// concatenation of all ext1's

    Writer* r;
    for (int i=0; i < this->writers.size(); i++)
	{
	r = this->writers.at(i);
	ext1 = r->getSupportedFileExtensions();
	for (int j = 0; j < ext1.size(); j++)
	    {
	    extAll.push_back(ext1.at(j));
	    } // for j
	} // for i
    return extAll;
}


QStringList Manager::getSupportedFileWriteExtensionsWithDescriptions()
{
	// List of extensions for one reader
	QStringList extensions;

	// List of file descriptions for one reader
	QStringList descriptions;

	// Combined output
	QStringList output;

	// One string containing all supported file types (default selection)
	QString allFiles = "All (";

	Writer * r;

	// Loop through all readers
	for (int i = 0; i < this->readers.size(); ++i)
	{
		// Get the current reader
		r = this->writers.at(i);

		// Get the supported extensions and descriptions
		extensions   = r->getSupportedFileExtensions();
		descriptions = r->getSupportedFileDescriptions();

		// Extension list and description list should have the same size. If you're
		// coding a new plugin and you triggered this assert, you have probably hard-
		// coded extension/description lists of different sizes.

		assert(extensions.size() == descriptions.size());
		
		// Loop through all supported file types for this reader
		for (int j = 0; j < extensions.size(); ++j)
		{
			// Create the combined output
			output.push_back(descriptions.at(j) + " (*." + extensions.at(j) + ")");

			// Append extension to the filter for all supported file types
			allFiles.append("*." + extensions.at(j) + " ");
		}
	}

	// Remove the last character (space), and add the closing parenthesis
	allFiles.resize(allFiles.length() - 1);
	allFiles.append(")");

	// Sort the list by alphabetic order
	output.sort();

	// Add the "allFiles" string to the front (default option)
	output.push_front(allFiles);

	return output;
}


void Manager::loadDataFromFile(QString filename)
{
    QTextStream out(stdout);
    out<<"Loading data from file "<<filename<<endl;
    QFile file;
    file.setFileName(filename);
    if (!file.exists())
	{
	out<<"File with filename "<<filename<<" does not exist!"<<endl;
	return;
	}

    Reader* reader = NULL;
    int i = 0;
    QStringList ext1;
    while ((reader == NULL) && i < this->readers.size())
	{
	ext1 = this->readers.at(i)->getSupportedFileExtensions();
	for (int j=0; j < ext1.size(); j++)
	    {
	    if (file.fileName().endsWith(QString(ext1.at(j))))
		{
		reader = this->readers.at(i);
		}
	    } // for j
	    i++;
	} // while

    if (!reader)
	{
	out << "No reader found that supports the requested file extension for file "<<filename<<endl;
	return;
	} // if
    
    reader->loadDataFromFile(filename);
}


void Manager::writeDataToFile(QString filename, DataSet *ds)  
{
    QTextStream out(stdout);
    QFile file;
    file.setFileName(filename);
     

    Writer* writer = NULL;
    int i = 0;
    QStringList ext1;
    while ((writer == NULL) && i < this->writers.size())
	{
	ext1 = this->writers.at(i)->getSupportedFileExtensions();
	for (int j=0; j < ext1.size(); j++)
	    {
	    if (file.fileName().endsWith(QString(ext1.at(j))))
		{
		writer = this->writers.at(i);
		}
	    } // for j
	    i++;
	} // while

    if (!writer)
	{
	out<<"No writer found that supports the requested file extension for file "<<filename<<endl;
	return;
	} // if
    writer->writeDataToFile(filename, ds); 
}


} // namespace data
}  //namespace bmia
