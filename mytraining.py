import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""具体训练参数设置"""

if __name__ == '__main__':

    lr = 5e-4
    batch_size = 16
    # batch_size = 8
    num_epochs = 70
    weight_decay = 0.0001
    lr_period = 10
    lr_decay = 0.5
    save_path = "Toy"
    
    male_net = myKit.get_net()
    female_net = myKit.get_net()
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, save_path=save_path)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    myKit.train_fn(net=male_net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, 
                   lr_period=lr_period, lr_decay=lr_decay, batch_size=batch_size, model_path="model_Toy.pth", record_path="RECORD_Toy.csv", save_path=save_path)
