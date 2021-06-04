import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval, one_hot
from tensorboardX import SummaryWriter
from tqdm import tqdm

'''Train Benchmark'''    
    
def get_args():
    parser = argparse.ArgumentParser()
    # Important Parameters
    parser.add_argument('--dataset', type=str, default='MiniImageNet', 
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB'])
    parser.add_argument('--model_type', type=str, default='LeadNet', 
                        choices = ['ProtoNet', 'LeadNet'])    
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])    
    parser.add_argument('--balance', type=float, default=1)  # the regularizer during meta-training           
    parser.add_argument('--temperature', type=float, default=1.0)        
    parser.add_argument('--temperature2', type=float, default=1.0)   
    
    # Task-Related Parameters
    parser.add_argument('--way', type=int, default=5)        
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)    
    parser.add_argument('--query', type=int, default=5) 
    
    # Optimization Parameters
    parser.add_argument('--warmup', type=int, default=0)    
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--step', type=int, default=20)    
    parser.add_argument('--init_weights', type=str, default=None)    
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--n_concept', type=int, default=64)      
    parser.add_argument('--num_tasks', type=int, default=1)    
    args = parser.parse_args()
    
    pprint(vars(args))
    set_gpu(args.gpu)
    save_path1 = '{}-{}-{}-w{}s{}-{}q{}'.format(args.dataset, args.model_type, args.backbone_class, args.way, args.shot, args.eval_shot, args.query)
    save_path2 = '_'.join([str(args.lr), str(args.step), str(args.gamma), str(args.warmup)])
    save_path2 += '-T1{}T2{}B{}C{}T{}'.format(args.temperature, args.temperature2, 
                                              args.balance, args.n_concept, args.num_tasks)
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)
    args.orig_imsize = -1
    args.eval_way = args.way
    return args

def get_model(args):
    if args.model_type == 'ProtoNet':
        from model.models.protonet import ProtoNet        
        model = ProtoNet(args)
    elif args.model_type == 'LeadNet':
        from model.models.LeadNet import LeadNet        
        model = MMap(args)                     
    else:
        raise ValueError('No Such Model')
        
    if args.init_weights is not None:
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights)['params']
        filter_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(filter_pretrained_dict.keys())
        model_dict.update(filter_pretrained_dict)
        model.load_state_dict(model_dict)    
        
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    return model

def get_optimizer(args, model):
    parameters = model.named_parameters()
    top_list, bottom_list = [], []
    for k, v in parameters:
        if 'encoder' in k:
            bottom_list.append(v)
        else:
            top_list.append(v)
            
    if args.warmup > 0:
        # prepare warmup optimizer
        optimizer_warmup = torch.optim.SGD(top_list, 
                                           lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer_warmup, lambda epoch: epoch * (1/args.warmup))           
    
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)  
        return optimizer, lr_scheduler, optimizer_warmup, lr_scheduler_warmup
    else:              
        optimizer = torch.optim.SGD([{'params': top_list, 'lr': args.lr * 10},
                                     {'params':bottom_list}],
                                    lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)  
        
        return optimizer, lr_scheduler, None, None
    
def get_loader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, num_workers=4, batch_sampler=train_sampler, pin_memory=True)
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 600, args.way, args.eval_shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    return train_loader, val_loader
    
def get_eval_loader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5
    else:
        raise ValueError('Non-supported Dataset.')
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 10000, args.way, args.eval_shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)   
    return test_loader
    
def get_cross_shot_dataloader(args, shot):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5
    else:
        raise ValueError('Non-supported Dataset.')
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 10000, args.way, shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)   
    return test_loader

def train_model(args, model, train_loader, val_loader):
    optimizer, lr_scheduler, optimizer_warmup, lr_scheduler_warmup = get_optimizer(args, model)
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['val_loss_interval'] = []    
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['val_acc_interval'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_interval'] = 0.0
    trlog['max_acc_epoch'] = 0
    
    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=args.save_path)
    
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        tl = Averager()
        ta = Averager()
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query).type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda() 
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, data_aug, gt_label = [_.cuda() for _ in batch]
            else:
                data, data_aug, gt_label = batch[0], batch[1], batch[2]
            logits, reg_loss = model(data, gt_label, data_aug)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label.view(-1))            
            if reg_loss is not None:
                loss = loss + args.balance * reg_loss
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)            
            if (i-1) % 50 == 0:
                print('epoch {}, train {}/{}, loss={:.4f}, acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), float(acc)))

            tl.add(loss.item())
            ta.add(acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del logits, loss
            torch.cuda.empty_cache()

        tl = tl.item()
        ta = ta.item()
        lr_scheduler.step()
        
        model.eval()
        results = np.zeros((600, 2))
        print('best epoch {}, best AVG val acc={:.4f}+{:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['max_acc_interval']))
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query).type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()      
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, data_aug, gt_label = [_.cuda() for _ in batch]
                else:
                    data, data_aug, gt_label = batch[0], batch[1], batch[2]
                logits, reg_loss = model(data, gt_label, data_aug)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label.view(-1)) 
                results[i-1, 0] = loss.item()
                results[i-1, 1] = acc
                
        mean_acc, interval = [], []
        for j in range(2):
            vl, vlp = compute_confidence_interval(results[:, j])
            mean_acc.append(vl)
            interval.append(vlp)
        
        writer.add_scalar('data/val_loss', float(mean_acc[0]), epoch)
        writer.add_scalar('data/val_acc', float(mean_acc[1]), epoch)       
        print('epoch {}, val acc={:.4f}+{:.4f}'.format(epoch, mean_acc[1], interval[1]))
    
    
        if mean_acc[1] > trlog['max_acc']:
            trlog['max_acc'] = mean_acc[1]
            trlog['max_acc_interval'] = interval[1]
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['val_loss'].append(mean_acc[0])
        trlog['val_loss_interval'].append(interval[0])
        trlog['val_acc'].append(mean_acc[1])
        trlog['val_acc_interval'].append(interval[0])
        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()
    print(args.save_path)
    return model
    

def test_model(args, model):
    test_loader = get_eval_loader(args)
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc.pth'))['params'])
    model.eval()
    results = np.zeros((10000, 2)) # 0 for the mean acc, 1-3 for accuracy for sine, linear, and quad
    print('Best Epoch {}, best AVG val acc={:.4f}+{:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['max_acc_interval']))
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query).type(torch.LongTensor)
    concept_label = torch.eye(args.way).repeat([args.eval_shot+args.query, 1]).type(torch.LongTensor)            
    if torch.cuda.is_available():
        label = label.cuda()  
        concept_label = concept_label.cuda()
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            if torch.cuda.is_available():
                data, data_aug, gt_label = [_.cuda() for _ in batch]
            else:
                data, data_aug, gt_label = batch[0], batch[1], batch[2]
            logits, reg_loss = model(data, gt_label, data_aug)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label.view(-1)) 
            results[i-1, 0] = loss.item()
            results[i-1, 1] = acc
            
    mean_acc, interval = [], []
    for j in range(2):
        vl, vlp = compute_confidence_interval(results[:, j])
        mean_acc.append(vl)
        interval.append(vlp)
    
    print('Test acc={:.4f}+{:.4f}'.format(mean_acc[1], interval[1]))
    
    trlog['test_acc'] = mean_acc[1]
    trlog['test_acc_interval'] = interval[1]
    with open(osp.join(args.save_path, '{}+{}'.format(trlog['test_acc'], trlog['test_acc_interval'])), 'w') as f:
        f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
            trlog['max_acc_epoch'],
            trlog['max_acc'],
            trlog['max_acc_interval']))
        f.write('Test acc={:.4f} + {:.4f}\n'.format(
            trlog['test_acc'],
            trlog['test_acc_interval']))                           
                
                
def evaluate_test_cross_shot(args, model):
    # evaluation mode
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc.pth'))['params'])
    model.eval()   
    # num_shots = [1, 5, 10, 20, 30, 50]
    num_shots = [1, 5]
    record = np.zeros((10000, len(num_shots))) # loss and acc
    label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            trlog['max_acc_epoch'],
            trlog['max_acc'],
            trlog['max_acc_interval']))
    for s_index, shot in enumerate(num_shots):
        test_loader = get_cross_shot_dataloader(args, shot)
        args.eval_shot = shot
        args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
        args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.query        
        for i, batch in tqdm(enumerate(test_loader, 1)):
            if torch.cuda.is_available():
                data, data_aug, gt_label = [_.cuda() for _ in batch]
            else:
                data, data_aug, gt_label = batch[0], batch[1], batch[2]
            with torch.no_grad():
                logits, reg_loss = model(data, gt_label, data_aug)                
            # loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, s_index] = acc
        assert(i == record.shape[0])

        va, vap = compute_confidence_interval(record[:,s_index])
        print('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))
        args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query

    with open(osp.join(args.save_path, '{}+{}-CrossShot'.format(va, vap)), 'w') as f:
        f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                trlog['max_acc_epoch'],
                    trlog['max_acc'],
                    trlog['max_acc_interval']))                
        for s_index, shot in enumerate(num_shots):
            va, vap = compute_confidence_interval(record[:,s_index])
            f.write('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))


if __name__ == '__main__':
    args = get_args()
    train_loader, val_loader = get_loader(args)
    model = get_model(args)
    model = train_model(args, model, train_loader, val_loader)
    # test_model(args, model)
    evaluate_test_cross_shot(args, model)
    
