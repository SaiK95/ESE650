This is the instructions file on how to run my codes. 

1. Please place all imu data into a folder "imu" inside my submission folder.
2. Please place all vicon data into a folder "Vicon" inside my submission folder.
3. Please place all camera data into a folder "cam" inside my submission folder.
4. In the file "Data Processing.py", on lines 10 and 11, please enter the file index of the imu and vicon file you want to process. For example,
to process "imuRaw8.mat", enter imu_file_index = 8
5. Run the file "Data Processing.py". This will load the data, process it and extract all information required to implement the UKF.
6. Run the file "ukf.py". This will take about 40 seconds to 120 seconds depending on the size of the data. It will output the final plot as well.
Blue indicates Vicon data, Red indicates UKF Output.
7. The results are saved in the results folder.

P.S: At the suggestion of Sakthivel Sivaraman, I used the transforms3d package to test my rot2quat, rot2euler functions.