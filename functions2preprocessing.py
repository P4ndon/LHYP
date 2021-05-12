import numpy as np
import torch
import torchvision
import pickle
import os
import sklearn
import sklearn.model_selection
from os.path import join as pjoin
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from matplotlib import pyplot as plt

def train_val_test_set(patients_data, xdata, ydata, studyID, train_set, val_set, test_set):
    #splitting the data    
    if os.path.exists('test.txt') and os.path.exists('train.txt') and os.path.exists('val.txt'):
        with open('test.txt') as f:
            content_test = f.readlines()
        content_test = [x.strip() for x in content_test]
        with open('train.txt') as f:
            content_train = f.readlines()
        content_train = [x.strip() for x in content_train]
        with open('val.txt') as f:
            content_val = f.readlines()
        content_val = [x.strip() for x in content_val]

        train_set = []
        val_set = []
        test_set = [] 
        for patient in patients_data:
            x=torch.cat((patient.dcm_sa, patient.dcm_la), 0)
            if patient.studyID in content_train:
                train_set.append([x, patient.pathology])
            elif patient.studyID in content_val:
                val_set.append([x, patient.pathology])
            elif patient.studyID in content_test:
                test_set.append([x, patient.pathology])        
        print('datasets already done')
    else:
        x_train, x_test, y_train, y_test, studyID_train, studyID_test = sklearn.model_selection.train_test_split(xdata, ydata, studyID, test_size=0.2, stratify=ydata)
        x_train, x_val, y_train, y_val, studyID_train, studyID_val = sklearn.model_selection.train_test_split(x_train, y_train, studyID_train, test_size=0.25, stratify=y_train)
        #creating output file
        file_test = open('test.txt', 'a')
        file_train = open('train.txt', 'a')
        file_val = open('val.txt', 'a')

        #saving the studyIDs        
        for i in studyID_test:
            file_test.write(str(i) + "\n") 
        file_test.close()
        for i in studyID_train:
            file_train.write(str(i) + "\n") 
        file_train.close()
        for i in studyID_val:
            file_val.write(str(i) + "\n") 
        file_val.close()
        print('datasets done')
        train_hcm = 0
        val_hcm = 0
        test_hcm = 0
        train_normal = 0
        val_normal = 0
        test_normal = 0
        train_other = 0
        val_other = 0
        test_other = 0
        for i in y_train:
            if i == 0:
                train_normal += 1
            elif i == 1:
                train_hcm += 1
            else:
                train_other += 1
        for i in y_val:
            if i == 0:
                val_normal += 1
            elif i == 1:
                val_hcm += 1
            else:
                val_other += 1
        for i in y_test:
            if i == 0:
                test_normal += 1
            elif i == 1:
                test_hcm += 1
            else:
                test_other += 1

        for i in range(len(x_train)): #creating train dataset
            #if x_train[i].shape == torch.Size([15, 150, 150]):
            data = [x_train[i], y_train[i]]
            train_set.append(data)
        for i in range(len(x_val)): #creating val dataset
            #if x_val[i].shape == torch.Size([15, 150, 150]):
            data = [x_val[i], y_val[i]]
            val_set.append(data)
        for i in range(len(x_test)): #creating test dataset
            data = [x_test[i], y_test[i]]
            test_set.append(data)
    
    return train_set, val_set, test_set 

def dcm2tensor(patient):
    other_pathology = ['EMF', 'Fabry', 'Amyloidosis', 'Aortastenosis']
    if patient.pathology in other_pathology:
        patient.dcm_sa = torch.zeros(6, 150, 150)
        patient.dcm_la = torch.zeros(9, 150, 150)
    else:
        #defineing the center of the countour rectangle
        y = [] 
        x = []
        for i in range(len(patient.contours)):
            for j in range(len(patient.contours[i])):
                y.append(patient.contours[i][j][0])
                x.append(patient.contours[i][j][1])
        y_center = (np.max(y) - np.min(y)) / 2 + np.min(y)
        x_center = (np.max(x) - np.min(x)) / 2 + np.min(x)
        
        #sa image preprocessing
        row_l = [0]
        row_h = []
        col_l = [0]
        col_h = []
        for img in range(len(patient.dcm_sa)):
            img_size = patient.dcm_sa[img].shape
            #cutting the image: 75 px to each directions
            for row in range(patient.dcm_sa[img].shape[0]):
                if (row < (y_center - 75)):
                    row_l.append(row)
                elif (row > (y_center + 75)):
                    row_h.append(row)
            for col in range(patient.dcm_sa[img].shape[1]):
                if (col < (x_center+1 - 75)):
                    col_l.append(col)
                elif (col > (x_center + 75)):
                    col_h.append(col)
            if row_h == []:
                row_h = [patient.dcm_sa[img].shape[0]]
            if col_h == []:
                col_h = [patient.dcm_sa[img].shape[1]]
            patient.dcm_sa[img] = patient.dcm_sa[img][np.max(row_l):(np.min(row_h)-1), np.max(col_l):(np.min(col_h)-1)] 
            #"padding" the image with 0 to get 150x150 px images
            if patient.dcm_sa[img].shape[0] < 150:
                padding_row = np.zeros((150-patient.dcm_sa[img].shape[0], patient.dcm_sa[img].shape[1])) 
                patient.dcm_sa[img] = np.append(patient.dcm_sa[img], padding_row, axis = 0)
            if patient.dcm_sa[img].shape[1] < 150:
                diff = (np.min(col_h)-np.max(col_l))
                padding_col = np.zeros((150, 150 - patient.dcm_sa[img].shape[1]))
                patient.dcm_sa[img] = np.append(patient.dcm_sa[img], padding_col, axis = 1)
            patient.dcm_sa[img] = patient.dcm_sa[img] / 255 #rescaling
            #if np.random.random_sample() > 0.95:
            #    plt.imshow(patient.dcm_sa[img])
            #    plt.show()
        #converting the images to tensors. size: 6x150x150
        if len(patient.dcm_sa) != 6:
            patient.dcm_sa.append(np.zeros((150,150)))

        patient.dcm_sa = torch.tensor(patient.dcm_sa)

        #la images 
        if len(patient.dcm_la) != 0:
            for img in range(len(patient.dcm_la)):
                length = len(patient.dcm_la[img])
                if length < 195 and length >= 150:
                    patient.dcm_la[img] = patient.dcm_la[img][(length-150):length, (length-150):length]
                elif length < 150:
                    a=patient.dcm_la[img].shape
                    padding_row = np.zeros(((150-length), length)) 
                    b=padding_row.shape
                    patient.dcm_la[img] = np.append(patient.dcm_la[img], padding_row, axis = 0)
                    padding_col = np.zeros((150, (150-length)))
                    patient.dcm_la[img] = np.append(patient.dcm_la[img], padding_col, axis = 1)
                else:
                    patient.dcm_la[img] = patient.dcm_la[img][45:195, 45:195] #cutting the centre of the images
                patient.dcm_la[img] = patient.dcm_la[img] / 255 #rescaling
                #if np.random.random_sample() > 0.95:
                #    plt.imshow(patient.dcm_la[img])
                #    plt.show()
            
            patient.dcm_la = torch.tensor(patient.dcm_la)    
        else:
            patient.dcm_la = torch.zeros(9, 150, 150) #if dcm_la is empty, set it to a black tensor with the same size
        
    return patient.dcm_sa, patient.dcm_la

def BMI_average(patients_data):
    BMI_mean = 0
    num = 0
    for patient in patients_data:
        if patient.BMI != None:
            num +=1
            BMI_mean += patient.BMI

    BMI_mean = BMI_mean / num #average BMI  
    return BMI_mean

def BMI2tensor(patient, BMI_mean):
    if patient.BMI == None:
        patient.BMI = BMI_mean
    patient.BMI = torch.tensor(patient.BMI)    
    return patient.BMI

def label2tensor(patient, hcm, normal, other):
    
    
    hcm_pathology = ['HCM']
    normal_pathology = ['U18_f', 'U18_m', 'Normal', 'adult_m_sport', 'adult_f_sport']
    other_pathology = ['EMF', 'Fabry', 'Amyloidosis', 'Aortastenosis']

    #collecting and modifying the different pathology types
    if patient.pathology  in normal_pathology:
        patient.pathology = 0
        normal += 1
    elif patient.pathology in other_pathology:
        patient.pathology = 0 #2 
        other += 1
    elif patient.pathology in hcm_pathology:
        patient.pathology = 1
        hcm += 1
    else:
        print('new type of pathology, please update the pathology lists with:' + patient.pathology)
    return patient.pathology, hcm, normal, other

def train_val_dataloader(train_set, val_set, batch_size, hcm, normal, other):
    train_weights = []
    for i in train_set:
        if i[1] == 1: # it picks the patients with hcm a bigger prob
            train_weights.append(1/(hcm/(hcm+normal))) #hcm
        elif i[1] == 0:
            train_weights.append(1/(normal/(hcm+normal))) #normal
        else:
            train_weights.append(0)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=batch_size, replacement=True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, drop_last=True)

    val_dataloader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
    return train_dataloader, val_dataloader

