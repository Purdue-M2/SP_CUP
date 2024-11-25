# SP_CUP

### 1. Load Data
If you are using the **cup** training dataset and validation dataset, please ensure to load the following files in your code:

- **cup_train.h5**: Contains the training dataset
- **cup_val.h5**: Contains the validation dataset
```
parser.add_argument('--train_datapath', default='cup_train.csv', type=str)
parser.add_argument("--test_datapath", type=str,
                        default='cup_val.csv', help="test data path")

    train_dataset = ImageDataset(
    hdf5_filename='cup_train.h5',
    labels_filename=args.train_datapath,
    dataset_name='train_features'
)


    test_dataset = ImageDataset(
    hdf5_filename='cup_val.h5',
    labels_filename=args.test_datapath,
    dataset_name='test_features'
)
```
If you are using the **AI-Face** training dataset and **CUP** validation dataset, please ensure to load the following files in your code:

- **ai_face_all_v1.h5**: Contains the AI-Face training dataset
- **cup_val.h5**: Contains the validation dataset
```
parser.add_argument('--train_datapath', default='all_ai_face_v1.csv', type=str)
parser.add_argument("--test_datapath", type=str,
                        default='cup_val.csv', help="test data path")

    train_dataset = ImageDataset(
    hdf5_filename='cup_train.h5',
    labels_filename=args.train_datapath,
    dataset_name='train_features'
)


    test_dataset = ImageDataset(
    hdf5_filename='cup_val.h5',
    labels_filename=args.test_datapath,
    dataset_name='test_features'
)
```

### 2. Train the model
#### CVaR Loss
* If you are using CVaR loss, change the loss type to **cvar**, and tune the **alpha** from [0.1, 0.9]
```python
python model_trainer(loss_type='cvar',alpha=0.7,gamma=0.8)
```

#### AUC Loss
* If you are using AUC loss, change the loss type to **auc**, and tune the **gamma** from [0.1, 0.9]
```python
python model_trainer(loss_type='auc',alpha=0.7,gamma=0.8)
```

#### VS Loss
* If you are using VS loss, change the loss type to **vs**.
```python
python model_trainer(loss_type='vs',alpha=0.7,gamma=0.8)
```
In VS loss you should tune hyperparameters:**iota_pos,iota_neg,Delta_pos,Delta_neg**. You can consider the iota [-1,1], delta [0.5,2].

```
elif loss_type == 'vs':
    vs_loss_fn = BinaryVSLoss(iota_pos=-0.05, iota_neg=0.05, Delta_pos=0.9, Delta_neg=1.1, weight=None)
    loss = vs_loss_fn(output,labels)
    return loss
```
