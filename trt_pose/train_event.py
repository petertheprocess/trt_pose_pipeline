import argparse
import torch
import torchvision
import os
import torch.optim
import tqdm
import apex.amp as amp
import time
import json
import pprint
import torch.nn.functional as F
from .event_frame_dataset import EventFrameDataset
from .models import MODELS
import wandb

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6

def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, epoch):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'epoch_%d.pth' % epoch)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, test_loss):
    with open(logfile, 'a+') as f:
        logline = '%d, %f, %f' % (epoch, train_loss, test_loss)
        print(logline)
        f.write(logline + '\n')
        
device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    logfile_path = args.config + '.log'
    
    checkpoint_dir = args.config + '.checkpoints'
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)
    
    # initial wandb
    wandb.init(project=config['wandb']['project'],
               name=config['wandb']['name'],
               config=config,
               dir=checkpoint_dir)
    
    
        
    # LOAD DATASETS    
    dataset_kwargs = config["dataset"]
    
    dataset = EventFrameDataset(**dataset_kwargs)
    # shuffle dataset and save indices train and test split
    dataset.split_dataset(
        train_fraction=config['train_fraction'],
        seed=config['seed']
    )
    total_length = len(dataset)
    print('Total dataset length: %d' % total_length)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [int(total_length * config['train_fraction']),
         total_length - int(total_length * config['train_fraction'])]
    )
    # save train and test indices into a split file
    with open(os.path.join(checkpoint_dir, 'train_test_split.json'), 'w') as f:
        json.dump({
            'train_indices': train_dataset.indices,
            'test_indices': test_dataset.indices
        }, f)

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)

    
    if "initial_state_dict" in config['model']:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))

    # frozen backbone
    for param in model[0].parameters():
        param.requires_grad = False
    print('Model %s created with %d parameters' % (config['model']['name'], sum(p.numel() for p in model.parameters())))
    
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
        
    for epoch in range(config["epochs"]):
        
        if str(epoch) in config['stdev_schedule']:
            stdev = config['stdev_schedule'][str(epoch)]
            print('Adjusting stdev to %f' % stdev)
            train_dataset.stdev = stdev
            test_dataset.stdev = stdev
            
        if str(epoch) in config['lr_schedule']:
            new_lr = config['lr_schedule'][str(epoch)]
            print('Adjusting learning rate to %f' % new_lr)
            set_lr(optimizer, new_lr)
        
        if epoch % config['checkpoints']['interval'] == 0:
            save_checkpoint(model, checkpoint_dir, epoch)
        
        train_loss = 0.0
        model = model.train()
        for data_batch in tqdm.tqdm(iter(train_loader)):
            # image = image.to(device)
            # cmap = cmap.to(device)
            # paf = paf.to(device)
            image = data_batch['event_frame'].to(device)
            paf = data_batch['paf'].to(device)
            cmap = data_batch['cmap'].to(device)
                        
            optimizer.zero_grad()
            cmap_out, paf_out = model(image)
            
            cmap_mse = torch.mean( (cmap_out - cmap)**2)
            paf_mse = torch.mean( (paf_out - paf)**2)
            
            loss = cmap_mse + paf_mse
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_loss += float(loss)
            
        train_loss /= len(train_loader)
        
        test_loss = 0.0
        model = model.eval()

        for data_batch in tqdm.tqdm(iter(test_loader)):
            image = data_batch['event_frame']
            paf = data_batch['paf']
            cmap = data_batch['cmap']
            with torch.no_grad():
                image = image.to(device)
                cmap = cmap.to(device)
                paf = paf.to(device)                
                cmap_out, paf_out = model(image)
                
                cmap_mse = torch.mean( (cmap_out - cmap)**2)
                paf_mse = torch.mean( (paf_out - paf)**2)

                loss = cmap_mse + paf_mse

                test_loss += float(loss)
        test_loss /= len(test_loader)
        
        write_log_entry(logfile_path, epoch, train_loss, test_loss)
        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epoch': epoch
        })

        print('Epoch %d, train loss: %f, test loss: %f' % (epoch, train_loss, test_loss))

        # eval 4 random images uplaod to wandb
        if epoch % config['eval']['interval'] == 0:
            model.eval()
            with torch.no_grad():
                for i in range(config['eval']['num_images']):
                    data_batch = next(iter(test_loader))
                    rgb = data_batch['rgb']
                    image = data_batch['event_frame'].to(device)
                    cmap = data_batch['cmap'].to(device)
                    paf = data_batch['paf'].to(device)
                    
                    cmap_out, paf_out = model(image)
                    
                    # upload to wandb
                    wandb.log({
                        'epoch': epoch,
                        'image': [wandb.Image(image[i].cpu(), caption='Image %d' % i) for i in range(len(image))],
                        'cmap': [wandb.Image(cmap_out[i].cpu(), caption='CMap %d' % i) for i in range(len(cmap_out))],
                        'paf': [wandb.Image(paf_out[i].cpu(), caption='PAF %d' % i) for i in range(len(paf_out))],
                        'rgb': [wandb.Image(rgb[i], caption='RGB %d' % i) for i in range(len(rgb))]
                    })
        
        
        
