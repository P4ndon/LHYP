import pydicom
import numpy as np
import pickle
from os.path import join as pjoin
import os
from data_preprocessing import data_collector 

def data2pickle(path_to_sample, output_folder):

    global patients_data
    patients_data = []
    patients_data = data_collector(path_to_sample, output_folder, patients_data)

    
    for patient in patients_data:
        filename = path_to_out + '\\' + patient.studyID
        out = open(filename + '.pickle', 'wb')
        pickle.dump(patient, out)
        out.close()


#d = os.path.dirname(__file__) #creating output folder, name: data
#p = r'{}/data'.format(d)
#try:
#    os.makedirs(p)
#except OSError:
#    pass

path_to_out = 'data' #output folder
path_to_sample = 'sample' #input folder

data2pickle(path_to_sample, path_to_out)



