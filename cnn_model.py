import torch
import torch.nn as nn
import numpy as np
import sklearn
import pytorch_lightning as pl
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
            nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=self.hparams["p"]),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=self.hparams["p"]),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=self.hparams["p"]),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout2d(p=self.hparams["p"]),
            
            nn.Flatten(), #128*9*9
            nn.Linear(10368, self.hparams["n_hidden"]),
            nn.ReLU(),
            nn.Dropout2d(p=self.hparams["p"]),            
            nn.Linear(self.hparams["n_hidden"], 2)
        )
    
    def forward(self, x):
        x = x.view(-1, 15, 150, 150)
        x = self.model(x.float())
        return x
    
    def training_step(self, batch, batch_idx):
        image, target = batch[0], batch[1] 
        # Perform a forward pass on the network with inputs
        out = self.forward(image)

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

        # Perform a forward pass on the network with inputs
        out = self.forward(image)

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
    
'''
def evaluate_model(model, dataloader, device):
    test_scores = []
    model.eval()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        targets_mask = targets >= 0
        test_scores.append(np.mean((preds.cpu() == targets.cpu())[targets_mask].numpy()))

    return np.mean(test_scores)

def run_epoch(model, optimizer, dataloader, loss_fcn, device, train):
        
    device = next(model.parameters()).device
    
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs= model(xb.unsqueeze(0))
            loss = loss_fcn(outputs[:,1], yb.float())
            top1 = torch.argmax(outputs, dim=1)
            ncorrect = torch.sum(top1 == yb)
            if train: 
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()
        
    epoch_loss /= len(dataloader.dataset)
    epoch_acc = evaluate_model(model, dataloader, device)
    return epoch_loss, epoch_acc

def fit(model, optimizer, train_dataloader, val_dataloader, loss_fcn, device, max_epochs, patience):
    
    best_acc = 0
    curr_patience = 0
    
    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(model, optimizer, train_dataloader, loss_fcn, device, train=True)
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, train loss: {train_loss:.2e}, accuracy: {train_acc * 100:.2f}%")
        val_loss, val_acc = run_epoch(model, None, val_dataloader, loss_fcn, device, train=False)
        print(f"Epoch {epoch + 1: >3}/{max_epochs}, val loss: {val_loss:.2e}, accuracy: {val_acc * 100:.2f}%")
        
        if val_acc >= best_acc:
            best_epoch=epoch
            best_acc=val_acc
            best_model_weights=copy.deepcopy(model.state_dict())
        
        if epoch - best_epoch >= patience:
            break
    
    model.load_state_dict(best_model_weights)



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

'''
