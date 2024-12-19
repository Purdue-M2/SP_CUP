import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import json
import random
import argparse
from scipy import optimize
# import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


import os


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

class FakeRealDataset(Dataset):
    def __init__(self, hdf5_filename, labels_filename=None, dataset_name='train_features'):
        self.hdf5_filename = hdf5_filename
        self.dataset_name = dataset_name
        self.labels_available = labels_filename is not None
       

        # Load HDF5 features
        self.hdf5_file = h5py.File(self.hdf5_filename, 'r')
        self.features = self.hdf5_file[self.dataset_name]

        # Load labels from CSV if provided
        if self.labels_available:
            self.data_frame = pd.read_csv(labels_filename)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get the extracted feature from HDF5 file
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        # If labels are available, get them from the CSV
        if self.labels_available:
            label = torch.tensor(self.data_frame.loc[idx, 'Target'],dtype=torch.float32)
            return feature, label
        else:
            return feature

class DetectorModel(nn.Module):
    def __init__(self):
        super(DetectorModel, self).__init__()

        dropout_rate = 0.335

        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(768, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )


        self.output_layer = nn.Linear(768, 1)

        nn.init.kaiming_uniform_(self.layers[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layers[4].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layers[8].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        # print(x.shape,'!!!!!!!!!!!!!!!!')
        features = self.layers(x)
        output = self.output_layer(features)
        return output,features


parser = argparse.ArgumentParser(description='')

parser.add_argument('--seed', default=8079, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for the scheduler')
parser.add_argument('--weight_decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--checkpoints_dir', default='checkpoints_realAs1', type=str)
parser.add_argument('--train_datapath', default='cup_train_realAS1.csv', type=str)
parser.add_argument('--train_batchsize', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)

#################################test##############################
parser.add_argument("--test_datapath", type=str,
                        default='cup_val_realAS1.csv', help="test data path")
parser.add_argument('--test_batchsize', default=32, type=int)


args = parser.parse_args()


# metrics_file_path = f"{args.checkpoints}/traing_log.txt"
os.makedirs(args.checkpoints_dir, exist_ok=True)

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def threshplus_tensor(x):
    y = x.clone()
    pros = torch.nn.ReLU()
    z = pros(y)
    return z

def search_func(losses, alpha):
    return lambda x: x + (1.0/alpha)*(threshplus_tensor(losses-x).mean().item())

def searched_lamda_loss(losses, searched_lamda, alpha):
    return searched_lamda + ((1.0/alpha)*torch.mean(threshplus_tensor(losses-searched_lamda))) 

def augment_features(features):
    # Helper Functions
    def hard_example_interpolation(z_i, hard_example, lambda_1):
        return z_i + lambda_1 * (hard_example - z_i)

    def hard_example_extrapolation(z_i, mean_latent, lambda_2):
        return z_i + lambda_2 * (z_i - mean_latent)

    def add_gaussian_noise(z_i, sigma, lambda_3):
        epsilon = torch.randn_like(z_i) * sigma
        return z_i + lambda_3 * epsilon

    def affine_transformation(z_i, angle, scale):
        # Create an affine transformation matrix
        theta = torch.tensor([
            [scale * torch.cos(angle), -scale * torch.sin(angle), 0],
            [scale * torch.sin(angle), scale * torch.cos(angle), 0]
        ]).to(z_i.device)

        # Apply the transformation (assume z_i is flattened into a 2D vector)
        z_i_flat = z_i.view(-1, 2)  # Reshape to 2D coordinates
        z_i_transformed = torch.mm(z_i_flat, theta.T)
        return z_i_transformed.view(z_i.shape)  # Reshape back to original

    def distance(z_i, z_j):
        return torch.norm(z_i - z_j)

    # all_fake_samples = torch.cat(feature_maps, dim=0)
    mean_latent = torch.mean(features, dim=0)

    # Identify the hardest example (farthest from the mean)
    distances = torch.tensor([distance(z, mean_latent) for z in features])
    hard_example = features[torch.argmax(distances)]

    # Augment the feature maps
    augmented_features = []
    for feature in features:
        # Choose a random augmentation
        augmentations = [
            # lambda z: hard_example_interpolation(z, hard_example, random.random()),
            # lambda z: hard_example_extrapolation(z, mean_latent, random.random()),
            lambda z: add_gaussian_noise(z, random.random(), random.random()),
            # lambda z: affine_transformation(z, random.uniform(-3.14, 3.14), random.uniform(0.8, 1.2))
        ]
        chosen_aug = random.choice(augmentations)
        # augmented = torch.stack([chosen_aug(z) for z in feature])
        # augmented_features.append(augmented)
        augmented_feature = chosen_aug(feature)  # Apply directly if `feature` is not iterable
        augmented_features.append(augmented_feature)
    return torch.stack(augmented_features)


class pAUCLoss(nn.Module):

    def __init__(self, k=False, reduction='mean', norm=False):
        super(pAUCLoss, self).__init__()
        self.reduction = reduction
        self.norm = norm
        self.k = k

    def forward(self, score_neg, score_pos):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1) # [n_pos, n_neg] 2d-tensor
        logistic_loss = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6)) # [n_pos, n_neg] 2d-tensor
        if self.k:
            logistic_loss = torch.topk(logistic_loss, self.k)[0]
        if self.reduction == 'mean':
            if self.norm:
                return logistic_loss.mean()
            else:
                return logistic_loss.sum() / (len(score_pos) * len(score_neg))
        elif self.reduction == 'sum':
            return logistic_loss.sum()
        elif self.reduction == 'none':
            return logistic_loss
        else:
            return
            
class BinaryVSLoss(nn.Module):

    def __init__(self, reduction, iota_pos=-0.05, iota_neg=0.05, Delta_pos=0.8, Delta_neg=1.2, weight=None ):
        super(BinaryVSLoss, self).__init__()
        # Initialize iota and Delta tensors on the default device
        # iota [-1,1], delta [0.5,2]
    
        # additive adjustments for negative and positive labels, respectively
        self.iota_list = torch.tensor([iota_neg, iota_pos], dtype=torch.float32)
        # multiplicative adjustments for negative and positive labels, respectively
        self.Delta_list = torch.tensor([Delta_neg, Delta_pos], dtype=torch.float32)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        device = x.device
        target = target.to(device)
        self.iota_list = self.iota_list.to(device)
        self.Delta_list = self.Delta_list.to(device)

        index = torch.zeros((x.shape[0], 2), dtype=torch.uint8,device=device)  # Shape (batch_size, 2)

        target = target.view(-1, 1)  # Reshape target to (batch_size, 1)
        x = x.view(-1, 1)

       
        index.scatter_(1, target.long(), 1)  # One-hot encode target
        index_float = index.type(torch.cuda.FloatTensor)

        #select the additive and multiplicative adjustments for each label
        batch_iota = torch.matmul(self.iota_list, index_float.t())
        batch_Delta = torch.matmul(self.Delta_list, index_float.t())

        batch_iota = batch_iota.view((-1, 1))
        batch_Delta = batch_Delta.view((-1, 1))
        # print(batch_iota.shape,batch_Delta.shape,'batch_iota.shape,batch_Delta.shape')
        output = x * batch_Delta - batch_iota

        if self.reduction == 'none':
            return F.binary_cross_entropy_with_logits(10 * output, target, weight=self.weight, reduction = self.reduction)
        
        if self.reduction == 'mean':
            return F.binary_cross_entropy_with_logits(10 * output, target, weight=self.weight)

        

def train_epoch(model, optimizer, scheduler, criterion, train_loader,loss_type,alpha,gamma,augment_features=None):
    model.train()
    total_loss_accumulated = 0.0  # Initialize total loss accumulated over the epoch
    total_batches = 0  # Keep track of the number of batches processed

    alpha_cvar = alpha
    #------------- P_AUC parameter-------------------#

    gamma_auc = gamma
    #------------- P_AUC parameter-------------------#

    def calculate_loss(output, labels, loss_type, criterion):
        loss_ce = criterion(output, labels)
        
        if loss_type == 'erm':
            return loss_ce.mean()

        elif loss_type == 'dag':
            chi_loss_np = search_func(loss_ce, alpha_cvar)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            loss = searched_lamda_loss(loss_ce, cutpt, alpha_cvar)
            return loss

        elif loss_type == 'auc':
            p_auc_loss_fn = pAUCLoss(k=False, reduction='mean', norm=False)
            # Separate positive and negative scores
            score_pos = output[labels == 1]  # Logits for positive samples
            score_neg = output[labels == 0]  # Logits for negative samples
            
            if len(score_pos) == 0 or len(score_neg) == 0:
                # To avoid errors if a batch has only one class
                return loss_ce.mean()
            
            auc_loss = p_auc_loss_fn(score_neg, score_pos)
            loss = loss_ce.mean() + gamma_auc * auc_loss
            return loss
        
        elif loss_type == 'vs':
            vs_loss_fn = BinaryVSLoss('mean', iota_pos=-0.05, iota_neg=0.05, Delta_pos=0.9, Delta_neg=1.1, weight=None)
            loss = vs_loss_fn(output,labels)
            return loss
        
        elif loss_type == 'vs_cvar_auc':
            vs_loss_fn = BinaryVSLoss('none', iota_pos=-0.05, iota_neg=0.05, Delta_pos=0.9, Delta_neg=1.1, weight=None)
            vs_loss = loss = vs_loss_fn(output,labels)
            chi_loss_np = search_func(vs_loss, alpha_cvar)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            cvar_loss = searched_lamda_loss(loss_ce, cutpt, alpha_cvar)

            p_auc_loss_fn = pAUCLoss(k=False, reduction='mean', norm=False)
            # Separate positive and negative scores
            score_pos = output[labels == 1]  # Logits for positive samples
            score_neg = output[labels == 0]  # Logits for negative samples
            
            if len(score_pos) == 0 or len(score_neg) == 0:
                # To avoid errors if a batch has only one class
                return loss_ce.mean()
            
            auc_loss = p_auc_loss_fn(score_neg, score_pos)
            loss = cvar_loss + gamma_auc * auc_loss
            return loss




    for batch in tqdm(train_loader, desc="Training Epoch"):
        inputs, target_labels = batch
        inputs, target_labels= inputs.to(device), target_labels.to(device)


        # Apply augmentation to all features
        if augment_features:
            augmented_inputs = augment_features(inputs)  # Augment all features

            # Combine original features with augmented features
            combined_inputs = torch.cat([inputs, augmented_inputs], dim=0)
            combined_labels = torch.cat([target_labels, target_labels], dim=0)  # Labels are duplicated

            # Shuffle the combined batch to avoid ordering effects
            perm = torch.randperm(combined_inputs.size(0))
            inputs, target_labels = combined_inputs[perm], combined_labels[perm]

        optimizer.zero_grad()
        output,_ = model(inputs)
        # loss_batch = criterion(output.squeeze(), target_labels)
        total_loss = calculate_loss(output.squeeze(), target_labels, loss_type, criterion)
       
        # total_loss = loss_batch 
        total_loss.backward()
        optimizer.step()

        # Accumulate the total loss and increment the batch count
        total_loss_accumulated += total_loss.item()
        total_batches += 1

    scheduler.step()

    average_loss = total_loss_accumulated / total_batches

    return average_loss


def calc_cm_each(matrix):
    ans = []
    for i in range(len(matrix)):
        ans.append(matrix[i][i] / np.sum(matrix[i]))

    return ans


def evaluate(model,  val_loader):
    model.eval()

    total_loss = 0
    total_correct = 0

    conf_matrix = np.zeros((2, 2))


    total_samples = 0

    all_predictions = []
    all_true_labels = []
    all_target_labels = []

    # To collect logits and features
    all_logits = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, target_labels = batch

            inputs, target_labels = inputs.to(device), target_labels.to(device)

            output, _ = model(inputs)

            # Collect the raw logits (before sigmoid)
            all_logits.extend(output.cpu().numpy())

            # Collect the input features (for pairwise distance calculation)
            all_features.extend(inputs.cpu().numpy())

            criterion = nn.BCEWithLogitsLoss()

            total_loss += criterion(output.squeeze(), target_labels).item()

            output = torch.sigmoid(output.squeeze())

            predicted = (output >= 0.5).float()

            total_correct += (predicted == target_labels).sum().item()
            total_samples += target_labels.size(0)

            conf_matrix += confusion_matrix(target_labels.cpu().numpy(), 
                                                predicted.cpu().numpy(),
                                                labels=[0, 1])

            # Accumulate the predictions, true labels, and gender labels
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(target_labels.cpu().numpy())
            all_target_labels.extend(target_labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        auc = roc_auc_score(all_true_labels, all_logits)

        # Precision, Recall, and F1 Score
        precision = precision_score(all_true_labels, all_predictions, labels=[0, 1], average="macro")
        recall = recall_score(all_true_labels, all_predictions, labels=[0, 1], average="macro")
        f1 = f1_score(all_true_labels, all_predictions, labels=[0, 1], average="macro")

       

        return (val_loss, accuracy, auc, precision, recall, f1)


def model_trainer(loss_type, alpha,gamma,aug):

    model = DetectorModel().to(device)
    train_dataset = FakeRealDataset(
    hdf5_filename='cup_train.h5',
    labels_filename=args.train_datapath,
    dataset_name='train_features'
)


    test_dataset = FakeRealDataset(
    hdf5_filename='cup_val.h5',
    labels_filename=args.test_datapath,
    dataset_name='test_features'
)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batchsize,shuffle=False,pin_memory=True)

  
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs / 4, eta_min=args.min_lr)  # eta_min is the minimum lr

    if loss_type == 'dag':
        args.checkpoints_dir=f'{args.checkpoints_dir}_{loss_type}_alpha_{alpha}_aug_{aug}'
    elif loss_type == 'auc':
        args.checkpoints_dir=f'{args.checkpoints_dir}_{loss_type}_gamma_{gamma}_aug_{aug}'
    elif loss_type =='vs':
        args.checkpoints_dir=f'{args.checkpoints_dir}_{loss_type}_aug_{aug}e'
    elif loss_type == 'erm':
        args.checkpoints_dir=f'{args.checkpoints_dir}_{loss_type}_aug_{aug}'
    elif loss_type == 'vs_cvar_auc':
        args.checkpoints_dir=f'{args.checkpoints_dir}_{loss_type}_alpha_{alpha}_gamma_{gamma}_aug_{aug}'

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    metrics_file_path = os.path.join(args.checkpoints_dir, 'performance_metrics.txt')
    print(metrics_file_path,'metrics_file_path')
    with open(os.path.join(args.checkpoints_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(metrics_file_path, 'w') as f:
        f.write('Epoch,Train Loss,Validation Loss,Validation Accuracy,Validation AUC,Validation Precision,Validation Recall,Validation F1 Score\n')

    for epoch in range(args.num_epochs):
        epoch_str = str(epoch).zfill(4)
        print(epoch_str)

        # Training phase
        train_loss = train_epoch(model, optimizer, scheduler, criterion, train_loader, loss_type,alpha,gamma,augment_features=augment_features)

        # Validation phase
        (val_loss, 
        accuracy, 
        auc, 
        precision, 
        recall, 
        f1_score) = evaluate(model, test_loader)

        # Logging metrics to console and metrics file
        metrics_str = (
            f'Epoch: {epoch_str}\n'
            f'Train Loss: {train_loss:.6f}\n'
            f'Validation Loss: {val_loss:.6f}\n'
            f'Validation Accuracy: {accuracy:.6f}\n'
            f'Validation AUC: {auc:.6f}\n'
            f'Validation Precision: {precision:.6f}\n'
            f'Validation Recall: {recall:.6f}\n'
            f'Validation F1 Score: {f1_score:.6f}\n\n'
        )
        print(metrics_str)
        with open(metrics_file_path, 'a') as f:
            f.write(f'{epoch_str}, {train_loss}, {val_loss}, {accuracy}, {auc}, {precision}, {recall}, {f1_score}\n')

        print()


        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoints_dir, f'checkpoint_epoch_{epoch}.pt')
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(checkpoint, checkpoint_path)

        # # Log metrics to result file
        # print(f'{epoch},'
        #     f'{val_loss:.6f},'
        #     f'{accuracy:.6f},'
        #     f'{auc:.6f},'
        #     f'{precision:.6f},'
        #     f'{recall:.6f},'
        #     f'{f1_score:.6f},'
        #     f'{target[0]},{target[1]}',
        #     file=fw, flush=True)



if __name__ == '__main__':
    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)



    model_trainer(loss_type='vs_cvar_auc',alpha=0.9,gamma=0.6, aug=True)
