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
