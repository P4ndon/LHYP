import matplotlib.pyplot as plt
import pydicom
import numpy as np
import pickle
from os.path import join as pjoin
import os
from pydicom.data import get_testdata_files
from dicom_reader import DCMreaderVM
from con_reader import CONreaderVM
from con2img import draw_contourmtcs2image as draw
from patient_class import patient
from matplotlib import pyplot as plt


def data_collector(path_to_sample, output_folder, pd):

    for folder_name in os.listdir(path_to_sample): #iteration in each patient folder
        actual_patient = patient()
        actual_patient.studyID = folder_name
        actual_patient.dcm = []
        print(folder_name)

        #collecting short axis dcm
        if os.path.isdir(pjoin(path_to_sample, folder_name, 'sa')) == True and len(os.listdir(pjoin(path_to_sample, folder_name, 'sa', 'images'))) != 0 :                  
            image_folder = pjoin(path_to_sample, folder_name, 'sa', 'images')
            dr = DCMreaderVM(image_folder)
            if dr.broken == False:
                number_of_images = len([f for f in os.listdir(image_folder)if os.path.isfile(os.path.join(image_folder, f))])
                mid = int(np.rint(number_of_images/25/2))  #selecting the middle slice
                slc = [mid-1, mid, mid+1]

                #collecting the contours
                con_file = pjoin(path_to_sample, folder_name, 'sa', 'contours.con')
                cr = CONreaderVM(con_file)
                contours = cr.get_hierarchical_contours()
            
                for slc in contours:
                    for frm in contours[slc]: 
                        image = dr.get_image(slc, frm)
                        #cutting-out the noise
                        if len(actual_patient.dcm) < 6 and isinstance(image, np.ndarray) == True:
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
                            actual_patient.dcm.append(image)

                #Patient_gender
                actual_patient.gender = cr.volume_data["Patient_gender="].split('\n')[0]

                #Patient_weight
                weight = float(cr.volume_data["Patient_weight="].split(' kg\n')[0])
                actual_patient.weight = weight

                #BMI [kg/m^2]
                #height is not always available
                vizsgalat = cr.volume_data["Study_description="].split()
                if cr.volume_data["Patient_height"] == None and ('cm' in cr.volume_data["Study_description="]):
                    height = float(cr.volume_data["Study_description="].split()[3])
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
                pd.append(actual_patient)
            
    return pd






