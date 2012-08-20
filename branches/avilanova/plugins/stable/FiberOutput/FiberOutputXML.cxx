/** 
 * FiberOutputXML.cxx
 *
 * 2008-01-28	Jasper Levink
 * - First Version.
 *
 * 2010-12-20	Evert van Aart
 * - First version for the DTITool3.
 *
 */


/** Includes */

#include "FiberOutputXML.h"


using namespace std;


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberOutputXML::FiberOutputXML() : FiberOutput()
{

}


//------------------------------[ Destructor ]-----------------------------\\

FiberOutputXML::~FiberOutputXML()
{

}


//------------------------------[ outputInit ]-----------------------------\\

void FiberOutputXML::outputInit() 
{
	// Open the output file
	this->outfile.open(this->fileName);

	// Check if the output file has been opened successfully
	if (!this->outfile.is_open())
		return;

	// Add the XML file header, Excel definition, and column header style definition
	this->outfile
		<< "<?xml version=\"1.0\"?>" << endl
		<< "<ss:Workbook xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\">" << endl
		<< "	<ss:Styles>" << endl
        << "		<ss:Style ss:ID=\"1\">" << endl
        << "			<ss:Font ss:Bold=\"1\"/>" << endl
        << "		</ss:Style>" << endl
		<< "	</ss:Styles>" << endl;
}


//----------------------------[ outputHeader ]-----------------------------\\

void FiberOutputXML::outputHeader() 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Initialize the header worksheet
	this->outputInitWorksheet("Header");

	// The header is just one column with several lines of text
	this->outfile
		<< "			<ss:Column ss:Width=\"250\"/>" << endl;

	this->outputWriteRow();
	this->outputWriteRow(&(string) "This is a DTI data file exported by DTITool");
	this->outputWriteRow(&(string) "This file is formatted for easy import in Microsoft Excel (r)");
	this->outputWriteRow();
	this->outputWriteRow(&(string) "More info about the DTITool:");
	this->outputWriteRow();
	this->outputWriteRow(&(string) "http://www.bmia.bmt.tue.nl/software/dtitool");
	this->outputWriteRow();
	this->outputWriteRow(&((string) "DTIfile: " + this->tensorImageInfo.name));
	this->outputWriteRow();

	// Print whether we use fibers or ROIs as the data source
	string strDataSource = "Datasource: ";

	if(this->dataSource == DS_ROI)
	{
		strDataSource += "ROI";
	} 
	else if(this->dataSource == DS_Fibers)
	{
		strDataSource += "Fiber"; 
	}

	this->outputWriteRow(&strDataSource);

	// Print the selected parameters
	string strParameters = "Parameters: ";

	for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
	{
		// Append measure name
		InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
		strParameters += currentInputInfo.name;

		if(currentMeasure < this->numberOfSelectedMeasures - 1) 
		{
			strParameters += ", ";
		}
	}

	this->outputWriteRow(&strParameters);

	// Print whether or not we use the "Per Voxel" settings
	string strPerVoxel = "Per Voxel: ";

	if(this->perVoxel) 
	{
		strPerVoxel += "yes"; 
	}
	else 
	{
		strPerVoxel += "no";
	}

	this->outputWriteRow(&strPerVoxel);

	// Print whether or not we the "Mean and Variance" setting
	string strMeanAndVar = "Mean and Variance: ";

	if(this->meanAndVar) 
	{
		strMeanAndVar += "yes";
	}
	else 
	{
		strMeanAndVar += "no";
	}

	this->outputWriteRow(&strMeanAndVar);

	// Finalize the header worksheet
	this->outputEndWorksheet();	
}


//-------------------------[ outputInitWorksheet ]-------------------------\\

void FiberOutputXML::outputInitWorksheet(string titel) 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Write the start of a new worksheet. Worksheet titel should be shortened to
	// at most 31 characters, and certain characters should be removed.

	this->outfile
		<< "	<ss:Worksheet ss:Name=\"" << this->removeReservedCharacters(this->shortenWorksheetName(titel)) << "\">" << endl
		<< "		<ss:Table>" << endl;
} 


//----------------------------[ outputWriteRow ]---------------------------\\

void FiberOutputXML::outputWriteRow(string * content, int contentSize, int styleID) 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Add the style ID to the row definition if defined
	if(styleID > 0)
	{
		this->outfile << "			<ss:Row ss:StyleID=\"" << styleID << "\">" << endl;
	}
	else
	{
		this->outfile << "			<ss:Row>" << endl;
	}
	
	// Write the cells
	for(int currentCell = 0; currentCell < contentSize; ++currentCell) 
	{
		this->outfile
			<< "			    <ss:Cell>" << endl
			<< "				<ss:Data ss:Type=\"String\">" << content[currentCell] << "</ss:Data>" << endl
			<< "				</ss:Cell>" << endl;		
	}

	// End the current row
	this->outfile << "			</ss:Row>" << endl;
} 


//----------------------------[ outputWriteRow ]---------------------------\\

void FiberOutputXML::outputWriteRow(double * content, int contentSize, string label, int styleID) 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Add the style ID to the row definition if defined
	if(styleID > 0)
	{
		this->outfile << "			<ss:Row ss:StyleID=\"" << styleID << "\">" << endl;
	}
	else
	{
		this->outfile << "			<ss:Row>" << endl;
	}

    // Add the label if defined
	if(label.length() > 0)
	{
		this->outfile
			<< "			    <ss:Cell>" << endl
			<< "				<ss:Data ss:Type=\"String\">" << label << "</ss:Data>" << endl
			<< "				</ss:Cell>" << endl;	
	}

	// Write the cells
	for(int currentCell = 0; currentCell < contentSize; ++currentCell) 
	{
		// Check for minus infinity values, since these cause errors in Excel
		stringstream ss;
		ss << content[currentCell];
		string contentString = ss.str();

		if (contentString == "1.#INF")
		{
			contentString = "0";
		}

		this->outfile
			<< "			    <ss:Cell>" << endl
			<< "				<ss:Data ss:Type=\"Number\">" << contentString << "</ss:Data>" << endl
			<< "				</ss:Cell>" << endl;		
	}

	// End the current row
	this->outfile << "			</ss:Row>" << endl;
} 


//--------------------------[ outputEndWorksheet ]-------------------------\\

void FiberOutputXML::outputEndWorksheet()
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Write the end of the worksheet
	this->outfile
		<< "		</ss:Table>" << endl
		<< "	</ss:Worksheet>" << endl;		
}


//------------------------------[ outputEnd ]------------------------------\\

void FiberOutputXML::outputEnd()
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Write the end of the workbook
	this->outfile << "</ss:Workbook>" << endl;

	// Close the output file
	this->outfile.close();	
}


//-------------------------[ shortenWorksheetName ]------------------------\\

string FiberOutputXML::shortenWorksheetName(string longName)
{
	// Do nothing if the length is already okay
	if (longName.length() < 31)
	{
		return longName;
	}

	// Copy the last 27 characters of the long name, and put "..." in front of them.
	string shortName = "...";
	shortName.append(longName.substr(longName.length() - 27, 27));
	return shortName;
}


//-----------------------[ removeReservedCharacters ]----------------------\\

string FiberOutputXML::removeReservedCharacters(string name) 
{
	// Create a string with reserved characters
	string reservedCharacters = "\\?*:/";
	unsigned int currentReservedCharacter;

	// Check for every character in "name" whether it is a reserved character
	for(unsigned int currentCharacterIndex = 0; currentCharacterIndex < name.length(); ++currentCharacterIndex) 
	{
		// Get the current character
		char currentCharacter = name.at(currentCharacterIndex);

		// True if we have found a reserved character
		bool found = false;

		// Index of reserved characters
		currentReservedCharacter = 0;

		while(!found && currentReservedCharacter < reservedCharacters.length()) 
		{
			// Check if the current character is reserved
			if(currentCharacter == (char)reservedCharacters.at(currentReservedCharacter)) 
			{
				// If so, replace it with an underscore
				name.replace(currentCharacterIndex,1,"_");
				found = 1;
			}

			currentReservedCharacter++;
		}
	}

	// Return the new filename
	return name;
}


} // namespace bmia