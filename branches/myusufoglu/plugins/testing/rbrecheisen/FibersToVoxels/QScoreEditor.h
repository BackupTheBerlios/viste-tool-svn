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

#ifndef __QScoreEditor_h
#define __QScoreEditor_h

#include <QtGui>

class QScoreEditor : public QDialog
{
public:

	QScoreEditor(QWidget * parent = 0) : QDialog(parent)
	{
		initLayout();
		initConnections();
		initSize();
	}

	virtual ~QScoreEditor()
	{
	}

private slots:

	void handleButtonNewTF()
	{
		bool ok;
		QString text = QInputDialog::getText(
			0, "New Transfer Function", "Transfer Function Name", 
			QLineEdit::Normal, "TF_new", &ok);
		if(!ok || text.isEmpty())
			return;
		
	}

	void handleButtonLoadTF();
	void handleButtonSaveTF();
	void handleButtonAddEntryBefore();
	void handleButtonAddEntryAfter();
	void handleButtonRemoveEntry();
	void handleButtonRemoveAllEntries();
	void handleComboTF(QString);
	void handleSliderOpacity(int);

private:

	void initLayout()
	{
		_buttonNewTF.setText("New...");
		_buttonLoadTF.setText("Load...");
		_buttonSaveTF.setText("Save...");
		_buttonAddEntryBefore.setText("Add Before");
		_buttonAddEntryAfter.setText("Add After");
		_buttonRemoveEntry.setText("Remove");
		_buttonRemoveAllEntries.setText("Remove All");
		_layout.addWidget(_buttonNewTF,            0, 1);
		_layout.addWidget(_buttonLoadTF,           0, 2);
		_layout.addWidget(_buttonSaveTF,           0, 3);
		_layout.addWidget(new QLabel("TF"),        1, 0);
		_layout.addWidget(_comboTF,                1, 1, 1, 3);
		_layout.addWidget(_buttonAddEntryBefore,   2, 0);
		_layout.addWidget(_listWidgetEntries,      2, 1, 4, 3);
		_layout.addWidget(_buttonAddEntryAfter,    3, 0);
		_layout.addWidget(_buttonRemoveEntry,      4, 0);
		_layout.addWidget(_buttonRemoveAllEntries, 5, 0);
		_layout.addWidget(new QLabel("Opacity"),   6, 0);
		_layout.addWidget(_sliderOpacity,          6, 1, 1, 3);
		_layout.setColumnStretch(1);
		setLayout(_layout);
	}

	void initConnections()
	{
		connect(_buttonNewTF,            SIGNAL(clicked()),                    this, SLOT(handleButtonNewTF()));
		connect(_buttonLoadTF,           SIGNAL(clicked()),                    this, SLOT(handleButtonLoadTF()));
		connect(_buttonSaveTF,           SIGNAL(clicked()),                    this, SLOT(handleButtonSaveTF()));
		connect(_buttonAddEntryBefore,   SIGNAL(clicked()),                    this, SLOT(handleButtonAddEntryBefore()));
		connect(_buttonAddEntryAfter,    SIGNAL(clicked()),                    this, SLOT(handleButtonAddEntryAfter()));
		connect(_buttonRemoveEntry,      SIGNAL(clicked()),                    this, SLOT(handleButtonRemoveEntry()));
		connect(_buttonRemoveAllEntries, SIGNAL(clicked()),                    this, SLOT(handleButtonRemoveAllEntries()));
		connect(_comboTF,                SIGNAL(currentIndexChanged(QString)), this, SLOT(handleComboTF(QString)));
		connect(_sliderOpacity,          SIGNAL(valueChanged(int)),            this, SLOT(handleSliderOpacity(int)));
	}

	void initSize()
	{
		resize(400, 400);
	}

	QGridLayout _layout;
	QPushButton _buttonNewTF;
	QPushButton _buttonLoadTF;
	QPushButton _buttonSaveTF;
	QPushButton _buttonAddEntryBefore;
	QPushButton _buttonAddEntryAfter;
	QPushButton _buttonRemoveEntry;
	QPushButton _buttonRemoveAllEntries;
	QComboBox   _comboTF;
	QListWidget _listWidgetEntries;
	QSlider     _sliderOpacity;

	QList<
};

#endif
