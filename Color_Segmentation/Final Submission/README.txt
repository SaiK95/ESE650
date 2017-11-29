This is the instructions file on how to run my codes. 

1. Place all test images in a folder called "Test images", within my folder, "Final Submission".
2. Run "single_gauss.py". This will take in all images in "Test images" as inputs and will output all regions in the image that has barrel intensity of red color. 
3. It will save all these images in another folder called "Detected barrels".
4. Run "dimensions of barrel_working.py". This will take the images in "Detected barrels" (output of single_gauss) as inputs.
5. It will return the exact location of the barrel, put a bounding box around it and save it in a new folder called "Final Output Images".
6. It will also print out height, width, distance of the barrel to the camera. 
7. It will also print out bottom left, top right and centroid coordinates of the barrel.
8. It will also save these parameters to a file called "dimensions.npy".
9. The file "barrel_values.npy" contains all the red pixel values that I obtained by hand labelling. The code I used for hand labelling is also attached.
10. The code I used for hand labelling is called "segment_from_training.py". This file is system dependent and will not execute on the graders' systems.
11. I'm also attaching a sources and references file along with this.