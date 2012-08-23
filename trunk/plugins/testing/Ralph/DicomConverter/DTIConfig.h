#ifndef __DTIConfig_h
#define __DTIConfig_h

#include <iostream>
#include <vector>

#include <string.h>
#include <stdio.h>

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTIConfig.h
//! \class  DTIConfig
//! \author Ralph Brecheisen
//! \brief  Reads configuration files consisting of key/value pairs.
//---------------------------------------------------------------------------
class DTIConfig
{
public:

	//---------------------------------------------------------------------------
	//! Constructor, creates a list of 2-element string arrays to store the 
	//! configuration key/value pairs. For each element in the list elem[0]
	//! contains the key and elem[1] contains the value.
	//---------------------------------------------------------------------------
	DTIConfig()
	{
		this->KeyValuePairs = new vector<char **>;
	}

	//---------------------------------------------------------------------------
	//! Destructor
	//---------------------------------------------------------------------------
	~DTIConfig()
	{
		if(this->KeyValuePairs)
			delete this->KeyValuePairs;
	}

	//---------------------------------------------------------------------------
	//! Loads a configuration file and stores the key/value pairs in the list
	//! if they contain no detectable errors.
	//---------------------------------------------------------------------------
	bool LoadFile(const char *fname)
	{
		const char *func = "DTIConfig::LoadFile";

		// Open configuration file. Quit if it can't be opened.
		FILE *f = fopen(fname, "rt");
		if(f == 0)
		{
			cout << func << ": Error opening file" << endl;
			return false;
		}

		char *line = new char[256];
		bool skip = false;
		int nrline = 0;

		// Read each line in the configuration file and check if it is correct
		// before addding it to the configuration repository.
		while(!feof(f))
		{
			memset(line, 0, 256);
			fgets(line, 256, f);
			if(ferror(f))
			{
				cout << func << ":Error reading line " << nrline << endl;
				return false;
			}

			nrline++;

			// Do error checking
			if(line[0] == '#' || line[0] == '\n' || line == "" || line[0] == '\0' || line[0] == '\t')
				continue;
			
			// Check if line contains equal sign
			char *first = strchr(line, '=');
			char *last = strrchr(line, '=');
			if(first == 0)
			{
				cout << func << ": Line " << nrline << " does not contain equal sign" << endl;
				continue;
			}

			// Check if line contains only a single equal sign
			if(first != last)
			{
				cout << func << ": Line " << nrline << " contains multiple equal signs" << endl;
				continue;
			}

			// Get key
			char *tmp = new char[256];
			int k = 0;
			int poseq = (int) (first - line);
			for(int i = 0; i < poseq; i++)
				if(line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
					tmp[k++] = line[i];
			tmp[k] = '\0';

			// Skip line if key is empty
			if(k == 0)
			{
				cout << func << ": Line " << nrline << " contains empty key" << endl;
				delete [] tmp;
				continue;
			}

			char *key = new char[(int)strlen(tmp)+1];
			strcpy(key, tmp);
			delete [] tmp;

			// Get value
			char *tmptmp = new char[256];
			k = 0;
			int len = (int) strlen(line);
			for(int i = (poseq+1); i < len; i++)
				if(line[i] != ' ' && line[i] != '\n' && line[i] != '\0')
					tmptmp[k++] = line[i];
			tmptmp[k] = '\0';

			if(k == 0)
			{
				cout << func << ": Line " << nrline << " contains empty value" << endl;
				delete [] tmptmp;
				continue;
			}

			char *value = new char[(int)strlen(tmptmp)+1];
			strcpy(value, tmptmp);
			delete [] tmptmp;

			// Add key/value pairs to repository
			char **pair = new char*[2];
			pair[0] = key;
			pair[1] = value;
			this->KeyValuePairs->push_back(pair);
		}

		fclose(f);
		return true;
	}

	//---------------------------------------------------------------------------
	//! Saves configuration file with the given name.
	//---------------------------------------------------------------------------
	bool SaveFile(const char *fname)
	{
		const char *func = "DTIConfig::SaveFile";
		cout << func << ": Function not implemented" << endl;
		return false;
	}

	//---------------------------------------------------------------------------
	//! Saves configuration file with the same name.
	//---------------------------------------------------------------------------
	bool SaveFile()
	{
		const char *func = "DTIConfig::SaveFile";
		cout << func << ": Function not implemented" << endl;
		return false;
	}

	//---------------------------------------------------------------------------
	//! Prints contents of the configuration file to given ostream. If none is
	//! provided stdout will we used.
	//---------------------------------------------------------------------------
	void PrintKeyValuePairs(ostream &out)
	{
		const char *func = "DTIConfig::PrintKeyValuePairs";

		// Check if the key/pair repository is not empty
		if(this->KeyValuePairs->size() == 0)
		{
			out << func << ": No key/values available" << endl;
			return;
		}

		// Print the key/value pairs
		vector<char **>::iterator iter;
		for(iter = this->KeyValuePairs->begin(); iter != this->KeyValuePairs->end(); iter++)
		{
			char **tmp = (*iter);
			out << tmp[0] << "=" << tmp[1] << endl;
		}
	}

	//---------------------------------------------------------------------------
	//! Returns value of the given key. If required=false and the value is not
	//! found, no error will be reported.
	//---------------------------------------------------------------------------
	char *GetKeyValue(const char *key)
	{
		const char *func = "DTIConfig::GetKeyValue";

		// Check if a valid key has been specified
		if(key == 0 || key == '\0' || key == "")
			return 0;

		// Check if the key/pair repository is not empty. There is no need
		// to check if the repository is NULL because it was created in the
		// constructor.
		if(this->KeyValuePairs->size() == 0)
			return 0;

		// Search for the first value that matches the given key
		vector<char **>::iterator iter;
		for(iter = this->KeyValuePairs->begin(); iter != this->KeyValuePairs->end(); iter++)
		{
			char **tmp = (*iter);
			if(strcmp(key, tmp[0]) == 0)
				return tmp[1];
		}

		return 0;
	}

private:

	// List of key/value pair (char[0] = key, char[1] = value)
	vector<char **> *KeyValuePairs;
};

#endif
