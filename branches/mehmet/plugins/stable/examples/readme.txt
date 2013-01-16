This directory includes 3 subdirectories with example plug-ins.

ConeVis: Visualization and Gui
PolyDataReader: Reader
PolyDataVis: Consumer, Visualization and Gui

Look at these example plug-ins for implementing your own plug-ins. ConeVis
shows how to implement a visualization and a GUI, without dealing with
data. PolyDataReader shows how to read data from a file, and how to make
the data that was read available for other plug-ins to use.

Finally, PolyDataVis is the most advanced plug-in of the three. It shows
how to get data from the data manager, how to visualize this data, and how
to let the user change the visualization parameters using a GUI.

Note: Also read tool/plugin/Plugin.h and tool/core/SimpleCoreInterface.h,
as well as all of the plugin interface headers of the interfaces that
are implemented by your plugin.

For using or creating data, also read tool/data/DataSet.h

