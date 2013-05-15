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

#include "ClusteringSettingsIO.h"
#include "ClusteringPlugin.h"

namespace bmia {

ClusteringSettingsIO::ClusteringSettingsIO()
{
	this->in.setFileName("ERROR");
	this->start = 0;
}

ClusteringSettingsIO::~ClusteringSettingsIO()
{
	if (this->in.isOpen())
		this->in.close();
}

void ClusteringSettingsIO::writeOutputFile(QTableWidget * table)
{
	QString fileName = QFileDialog::getSaveFileName(NULL, "Write Bundle Settings", "", "Bundle Settings (*.bun)");

	QFile out(fileName);
	if (!out.open(QIODevice::WriteOnly | QIODevice::Text))
         return;

	QTextStream outstream(&out);

	int numberOfClusters = table->rowCount();

	for (int i = 0; i < numberOfClusters; ++i)
	{
		QComboBox * CB = (QComboBox *) table->cellWidget(i, 1);

		if (!CB)
		{
			outstream << "ERROR: No QComboBox defined in row " << i << "!\n";
			continue;
		}

		QString bundle = CB->currentText();

		outstream << i << " " << bundle << "\n";
	}

	out.flush();
	out.close();
}

bool ClusteringSettingsIO::openFileForReading()
{
	QString fileName = QFileDialog::getOpenFileName(NULL, "Read Bundle Settings", "", "Bundle Settings (*.bun)");
	
	this->in.setFileName(fileName);

	if (!this->in.open(QIODevice::ReadOnly | QIODevice::Text))
         return false;

	this->instream.setDevice(&(this->in));

	this->start = this->instream.pos();

	return true;
}

bool ClusteringSettingsIO::checkClusteringIDs(int maxInputID)
{
	this->instream.seek(this->start);
	int ID;
	QString name;

	while (!(this->instream.atEnd()))
	{
		this->instream >> ID;
		this->instream.skipWhiteSpace();
		name = this->instream.readLine();

		if (ID < 0 || ID >= maxInputID)
			return false;
	}

	return true;
}

void ClusteringSettingsIO::populateOutputClusters(ClusteringPlugin * plugin)
{
	this->instream.seek(this->start);
	int ID;
	QString name;

	while (!(this->instream.atEnd()))
	{
		this->instream >> ID;
		this->instream.skipWhiteSpace();
		name = this->instream.readLine();

		plugin->addOutputClusterFromFile(name);
	}
}

void ClusteringSettingsIO::setOutputClusters(QTableWidget * table)
{
	this->instream.seek(this->start);
	int ID;
	QString name;

	while (!(this->instream.atEnd()))
	{
		this->instream >> ID;
		this->instream.skipWhiteSpace();
		name = this->instream.readLine();

		QComboBox * CB = (QComboBox *) table->cellWidget(ID, 1);

		if (!CB)
		{
			continue;
		}

		int outID = CB->findText(name);

		CB->setCurrentIndex(outID);
	}	
}


}