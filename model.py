import torch
import torch.nn as nn
import numpy as np
import sklearn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics import F1, Accuracy, Recall, AveragePrecision

class HCM_Model(pl.LightningModule):
    
    def __init__(self, hparams):
        super(HCM_Model, self).__init__()
        self.hparams = hparams
        self.f1 = F1(2)
        self.accuracy = Accuracy()
        self.recall = Recall()
        #self.precision = AveragePrecision()

        self.model = nn.Sequential(
            Lambda(lambda x: x.view(-1, 15, 150, 150)),
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualStack( in_channels=16, out_channels=16, stride=2, num_blocks=self.hparams["n"]),
            ResidualStack( in_channels=16, out_channels=32, stride=2, num_blocks=self.hparams["n"]),
            ResidualStack( in_channels=32, out_channels=64, stride=2, num_blocks=self.hparams["n"]),
            nn.AdaptiveAvgPool2d(1),
            Lambda(lambda x: x.squeeze()),
            nn.Linear(64, 2),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )
    
    
    def training_step(self, batch, batch_idx):
        image, target = batch[0], batch[1] 
        image = image.float()
        # Perform a forward pass on the network with inputs
        out = self.model(image)

        # calculate the loss with the network predictions and ground truth targets
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out[:,1].float(), target.float())
        loss.backward(retain_graph=True)
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        #acc = preds.eq(target).sum().float() / target.size(0)

        # Log the accuracy and loss values to the tensorboard
        #self.log('acc', acc)
        # log step metric
        f1_score = self.f1(preds, target)
        train_acc =  self.accuracy(preds, target)
        #train_prec = self.precision(preds, target)
        #train_rec = self.recall(preds, target)
        
        return {'loss': loss, 'f1_score': f1_score, 'train_acc': train_acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1_score'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        #avg_prec = torch.stack([x['train_prec'] for x in outputs]).mean()
        #avg_rec = torch.stack([x['train_rec'] for x in outputs]).mean()
        self.log('train_epoch_loss', avg_loss)
        self.log('train_epoch_f1', avg_f1)
        self.log('train_epoch_acc', avg_acc)
        #self.log('train_epoch_precision', avg_prec)
        #self.log('train_epoch_recall', avg_rec)

    def validation_step(self, batch, batch_idx):
        image, target = batch[0], batch[1]
        image = image.float()
        # Perform a forward pass on the network with inputs
        out = self.model(image)

        # calculate the loss with the network predictions and ground truth targets
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(out[:,1].float(), target.float())
        
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        #acc = preds.eq(target).sum().float() / target.size(0)
        acc = self.accuracy(preds.float() , target)
        f1_score = self.f1(preds.float() , target)
        #prec = self.precision(preds.float() , target)
        #rec = self.recall(preds.float() , target)

        #self.log('val_loss', loss.item())

        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1_score}

    def validation_epoch_end(self, outputs):

        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        #avg_prec = torch.stack([x['val_prec'] for x in outputs]).mean()
        #avg_rec = torch.stack([x['val_rec'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        self.log('val_end_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.log('val_f1', avg_f1)
        #self.log('val_precision', avg_prec)
        #self.log('val_recall', avg_rec)
        self.log('val_acc_epoch_f1', self.f1.compute())
        self.log('val_accuracy', self.accuracy.compute())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        return optim
    

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],(0, 0, 0, 0, 0, out_channels - in_channels), mode="constant", value=0))
        else:
            self.skip = nn.Sequential()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        xb = F.relu(self.conv1_bn(self.conv1(input)))
        z = self.conv2(xb) + self.skip(input)
        xb = F.relu(self.conv2_bn(z))
        return xb # [batch_size, num_classes]

class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        self.resblocks = nn.ModuleList()
        self.resblocks.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(num_blocks-1):
            self.resblocks.append(ResidualBlock(out_channels, out_channels, stride=1))

    def forward(self, input):
        num_blocks = len(self.resblocks)
        for i in range(num_blocks):
            input = self.resblocks[i](input)
        # print(input.shape)
        return input

def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
