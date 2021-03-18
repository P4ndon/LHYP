import matplotlib.pyplot as plt
import pydicom
import numpy as np
import pickle
from os.path import join as pjoin
import os
from pydicom.data import get_testdata_files
from pydicom.pixel_data_handlers.util import apply_modality_lut
from dicom_reader import DCMreaderVM
from con_reader import CONreaderVM
from patient_class import patient
from matplotlib import pyplot as plt


def data_collector(path_to_sample, output_folder, pd):

    for folder_name in os.listdir(path_to_sample): #iteration in each patient folder
        actual_patient = patient()
        actual_patient.studyID = folder_name
        actual_patient.dcm_sa = []
        actual_patient.dcm_la = []
        #print(folder_name)

        #collecting short axis dcm_sa
        if os.path.isdir(pjoin(path_to_sample, folder_name, 'sa')) == True and len(os.listdir(pjoin(path_to_sample, folder_name, 'sa', 'images'))) != 0 :                  
            image_folder = pjoin(path_to_sample, folder_name, 'sa', 'images')
            dr = DCMreaderVM(image_folder)
            if dr.broken == False:
                number_of_images = len([f for f in os.listdir(image_folder)if os.path.isfile(os.path.join(image_folder, f))])
                mid = int(np.rint(number_of_images/25/2))  #selecting the middle slice
                slc_idx = [mid-1, mid, mid+1, mid+2, mid+3]

                #collecting the contours and dcm_sa images
                con_file = pjoin(path_to_sample, folder_name, 'sa', 'contours.con')
                cr = CONreaderVM(con_file)
                contours = cr.get_hierarchical_contours()
                for slc in slc_idx:
                    for frm in contours[slc]: 
                        image = dr.get_image(slc, frm)
                        #cutting-out the noise
                        if len(actual_patient.dcm_sa) < 6 and isinstance(image, np.ndarray) == True:
                            cntrs = []
                            for mode in contours[slc][frm]:
                                cntrs.append(contours[slc][frm][mode])
                                actual_patient.contours.append(contours[slc][frm][mode])
                            
                            p1, p99 = np.percentile(image, (1, 99))
                            image[image < p1] = p1
                            image[image > p99] = p99           
                            image = (image-np.amin(image))/(np.amax(image)-np.amin(image))*255 #rescaling
                            image = image.astype(np.uint8) #converting from float32 to uint8
                            #plt.imshow(image)
                            #plt.show()
                            actual_patient.dcm_sa.append(image)

                #Patient_gender
                actual_patient.gender = cr.volume_data["Patient_gender="].split('\n')[0]

                #Patient_weight
                weight = float(cr.volume_data["Patient_weight="].split(' kg\n')[0])
                actual_patient.weight = weight

                #BMI [kg/m^2]
                #height is not always available
                vizsgalat = cr.volume_data["Study_description="].split()
                height = None
                if cr.volume_data["Patient_height"] == None and (len(cr.volume_data["Study_description="].split()) >=3):
                    for idx, text in enumerate(cr.volume_data["Study_description="].split()):
                        if text[0] =='1' or text[0] == '2':
                            height = float(cr.volume_data["Study_description="].split()[idx])
                    actual_patient.height = height
                    
                elif 'cm' not in cr.volume_data["Study_description="] and cr.volume_data["Patient_height"] != None:
                    height = float(cr.volume_data["Patient_height"].split()[0].split('=')[1])
                    actual_patient.height = height
                else:
                    height = None
                    actual_patient.height = height
                
                if height != None:
                    actual_patient.BMI = round(weight / (height/100 * height/100),2)
                
                #patology
                meta = open(path_to_sample + '\\' + folder_name + '\meta.txt', 'rt')
                data = meta.readline()
                actual_patient.pathology = data.split("Pathology: ")[1].split(' \n')[0]
                meta.close()

                # collecting la dcm imges in 2, 4, LVOT order (only if all axes are present)
                if os.path.isdir(pjoin(path_to_sample, folder_name, 'la')) == True and len(os.listdir(pjoin(path_to_sample, folder_name, 'la'))) > 74:                  
                    image_folder_la = pjoin(path_to_sample, folder_name, 'la')
                    dcm_files = sorted(os.listdir(image_folder_la))
                    if len(os.listdir(pjoin(path_to_sample, folder_name, 'la'))) == 150: # if a 2. measurement was taken
                        frm_numbers = [78, 86, 94, 103, 111, 119, 128, 136, 144] #3 frames in each perspectives
                    else: 
                        frm_numbers = [3, 11, 19, 28, 36, 44, 53, 61, 69] #3 frames in each perspectives
                    
                    for file in frm_numbers:
                        if dcm_files[file].find('.dcm') != -1:
                            temp_ds = pydicom.dcmread(os.path.join(image_folder_la, dcm_files[file]))
                            temp_ds = apply_modality_lut(temp_ds.pixel_array, temp_ds)
                            p1, p99 = np.percentile(temp_ds, (1, 99))
                            temp_ds[temp_ds < p1] = p1
                            temp_ds[temp_ds > p99] = p99
                            temp_ds = (temp_ds-np.amin(temp_ds))/(np.amax(temp_ds)-np.amin(temp_ds))*255 #rescaling
                            temp_ds = temp_ds.astype(np.uint8)
                            #plt.imshow(temp_ds)
                            #plt.show()
                            actual_patient.dcm_la.append(temp_ds)

                pd.append(actual_patient)
            
    return pd






