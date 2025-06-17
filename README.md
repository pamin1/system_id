# System Identification Pipeline for Ackermann Drive Vehicles
### Requirements
Need a bag file containing ground truth odometry and ackermann drive topics

## Use Case
Modify scripts/bag_reader.py to use your bag file (db3)
Run this twice: once for odom, once for drive topic.
* **You must update the topics and fieldnames in the main function for the csv to export correctly**
  
With the topic CSVs created, move on to the data_collection file. Update the filenames and run this script.
Should result in plots and another saved CSV.

Move on to the bicycle_model file, update the csv_path in the main function, run this.
Wait while it optimizes the Cf and Cr parameters. Once complete, it will produce a plot of the resulting simulated trajectory.
