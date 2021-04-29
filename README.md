# Curacel_Work

To run the notebook properly, set the version of keras and tensorflow to be in accordance with the version in the requirements.txt

Run the notebook to understand the work process, it is best if your system has a GPU.

The python script is CarDamageDetection.py (please review), which will still be updated to fit our specification.

To run the script using CMD
 
 # training the model on coco weights
 python CarDamageDetection.py train --dataset=/path/to/datasetfolder --weights=coco

# testing the model
python CarDamageDetection.py mask --image=/path/to/image
