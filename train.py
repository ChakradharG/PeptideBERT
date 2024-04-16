import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


def train_model():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {curr_lr}')
        val_loss, val_acc = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}\tValidation Accuracy: {val_acc}\n')
        scheduler.step(val_acc)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss, 
                'val_loss': val_loss, 
                'val_accuracy': val_acc, 
                'lr': curr_lr
            })

        if val_acc >= best_acc and not config['debug']:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'acc': val_acc, 
                'lr': curr_lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    wandb.finish()


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

train_data_loader, val_data_loader, test_data_loader = load_data(config)
config['sch']['steps'] = len(train_data_loader)

model = create_model(config)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

if not config['debug']:
    run_name = f'{config["task"]}-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='PeptideBERT', name=run_name)

    save_dir = f'./checkpoints/{run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('./model/network.py', f'{save_dir}/network.py')

train_model()
if not config['debug']:
    model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)
test_acc = test(model, test_data_loader, device)
print(f'Test Accuracy: {test_acc}%')
