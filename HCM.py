import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random2 as random
from os.path import join as pjoin
from LHYP.patient_class import patient
from functions2preprocessing import dcm2tensor, BMI_average, BMI2tensor, label2tensor, train_val_test_set, train_val_dataloader
from model import HCM_Model, initialize_weight
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')
writer2 = SummaryWriter('runs2')


path = pjoin('banna_data') #path to pickles  LHYP/data
patients_data = [] #list to loaded pickles
#variables to count each pathology types
hcm = 0
normal = 0
other = 0

#loading pickles
for pck in os.listdir(path):
    infile = open(pjoin(path, pck), 'rb')
    actual_patient = pickle.load(infile)
    patients_data.append(actual_patient)
    infile.close()
print('unpickling done')

# creating train, val, test datasets
xdata = []
ydata = []
studyID = []
all_sa = []
all_la = []

pathology_types = {
        'HCM': 0,
        'U18_f': 0,
        'U18_m': 0,
        'Normal': 0,
        'adult_m_sport': 0,
        'adult_f_sport': 0,
        'EMF': 0,
        'Fabry': 0,
        'Amyloidosis': 0,
        'Aortastenosis': 0
    }
BMI_types = {
    'Underweight': 0,
    'Normal weight': 0,
    'Overweight': 0,
    'Obesity': 0,
    'No info': 0
}

BMI_mean = BMI_average(patients_data) 

for patient in patients_data:
    pathology_types[patient.pathology] +=1
    #pathology preprocessing
    patient.pathology, hcm, normal, other = label2tensor(patient, hcm, normal, other)

    # sa and la to tensor 
    patient.dcm_sa, patient.dcm_la = dcm2tensor(patient)

    if patient.BMI == None:
        BMI_types['No info'] +=1
    elif patient.BMI < 20:
        BMI_types['Underweight'] +=1
    elif (patient.BMI < 25) and (patient.BMI > 20):
        BMI_types['Normal weight'] +=1
    elif (patient.BMI < 30) and (patient.BMI > 25):
        BMI_types['Overweight'] +=1
    elif patient.BMI > 30:
        BMI_types['Overweight'] +=1
    # BMI to tensor (where BMI is not known setting it to the average value)
    patient.BMI = BMI2tensor(patient, BMI_mean)

    all_sa.append(patient.dcm_sa)
    all_la.append(patient.dcm_la)
    #creating
    xdata.append(torch.cat((patient.dcm_sa, patient.dcm_la), 0))
    ydata.append(patient.pathology)
    studyID.append(patient.studyID)



print('data preprocessing done')

batch_size= 64
train_set=[]
val_set = []
test_set=[]
train_set, val_set, test_set = train_val_test_set(patients_data, xdata, ydata, studyID, train_set, val_set, test_set)
train_dataloader, val_dataloader = train_val_dataloader(train_set, val_set, batch_size, hcm, normal, other)

k_folds = 5
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

for i, item in enumerate(train_dataloader):
    print('Batch {}'.format(i))
    images, label = item
    print(f"Datatype of Image: {type(images)}")
    print(f"Shape of the Image: {images.shape}")
    print(f"Label Values: {label}")

    if i+1 >= 1:
        break

for i in range(15):
    hparams = {    
        "learning_rate": random.uniform(1e-6, 1e-2), #1e-3,
        "n_hidden": random.randint(200, 700),
        "p": random.uniform(0.2, 0.6),
        "n": random.randint(2, 8),
    }
    print('Run:'+ str(i))
    print('learning rate:'+ str(hparams['learning_rate']))
    print('n_hidden:'+ str(hparams['n_hidden']))
    print('p:'+ str(hparams['p']))
    print('n:'+ str(hparams['n']))
    model = HCM_Model(hparams = hparams)
    model.apply(initialize_weight)

    optim = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    
    early_stop_callback = EarlyStopping(
    monitor='val_acc',
    min_delta=0.0,
    patience=30,
    verbose=False,
    mode='max' 
    )
    trainer = pl.Trainer(
        weights_summary=None,
        max_epochs=50,
        progress_bar_refresh_rate=25, 
        callbacks=[early_stop_callback],
        gpus=1 
    )
    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()
    #writer.add_graph(model.cpu(), images)
    
    trainer.fit(model, train_dataloader=train_dataloader,val_dataloaders=val_dataloader)

    writer.close()

print('done')
