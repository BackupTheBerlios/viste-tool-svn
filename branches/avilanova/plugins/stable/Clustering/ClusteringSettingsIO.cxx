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