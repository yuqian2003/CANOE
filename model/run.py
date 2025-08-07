import argparse,os,time,torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch.nn as nn
from dataloader import MyDataset
from framework_mp import MyModel_mp
from framework_tc import MyModel_tc
from tools import get_config, run_test, train_epoch, get_mapper, update_config, custom_collate
def listmle_loss(scores, labels):
    return torch.nn.functional.cross_entropy(scores, labels, reduction='mean')
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset', type=str, default='TC')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--dim', type=int, default=16, help='must be a multiple of 4')
parser.add_argument('--topic', type=int, default=400, help='LDA topic num')
parser.add_argument('--at', type=str, default='none', help='arrival time module type')
parser.add_argument('--encoder', type=str, default='trans', help='encoder type')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--epoch', type=int, default=100, help='epoch num')
parser.add_argument('--path', type=str, default="MP11111_1", help='mclp/data/xxxx')
parser.add_argument('--type', type=int, default=1, help='oscillator type (1-6)')
parser.add_argument('--bandwidth', type=float, default=None, help='bandwidth parameter for embedding')
args = parser.parse_args()
gpu_list = args.gpu
torch.manual_seed(args.seed)
dataset_path = f'./data/{args.dataset}'
timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
save_path = "TC"  
save_dir = f"./saved_models/{args.dataset}"
config_path = f"{save_dir}/settings.yml"
device = torch.device("cuda")
test_only = args.test
if __name__ == '__main__':
    get_mapper(dataset_path=dataset_path)

    update_config(config_path, key_list=['Embedding', 'base_dim'], value=args.dim)
    update_config(config_path, key_list=['Encoder', 'encoder_type'], value=args.encoder)

    #### dataset
    update_config(config_path, key_list=['Dataset', 'topic_num'], value=args.topic)

    #### model
    update_config(config_path, key_list=['Model', 'seed'], value=args.seed)
    update_config(config_path, key_list=['Model', 'at_type'], value=args.at)
    update_config(config_path, key_list=['Model', 'batch_size'], value=args.batch)
    update_config(config_path, key_list=['Model', 'epoch'], value=args.epoch)

    
    config = get_config(config_path, easy=True)

    dataset = MyDataset(config=config, dataset_path=dataset_path, device=device, load_mode='train')
    batch_size = config.Model.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                            collate_fn=lambda batch: custom_collate(batch, device, config))
    if args.dataset == "TC":
        model = MyModel_tc(config, osc_type=args.type, bandwidth=args.bandwidth)
    else: 
        model = MyModel_mp(config, osc_type=args.type, bandwidth=args.bandwidth)
    model.to(device)
    ce_time = torch.nn.CrossEntropyLoss(reduction='mean')
    lambda_time = getattr(config.Model, 'lambda_time', 0.4)
    lambda_loc = getattr(config.Model, 'lambda_loc', 0.9)
    lambda_rank = getattr(config.Model, 'lambda_rank', 0.6)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * batch_size} | Total trainable parameters: {total_params}")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        #lr=config.Adam_optimizer.initial_lr,
        lr = 0.005,
        betas=(0.9, 0.999), eps=1e-08, 
        weight_decay=0.01, amsgrad=False,
        maximize=False, foreach=None, capturable=False, 
        differentiable=False, fused=None
    )

    print(f"Dataset: {args.dataset} | Device: {device} | Model: {config.Encoder.encoder_type}")
    print(f"AT type: {config.Model.at_type} | topic_num: {config.Dataset.topic_num} | dim: {config.Embedding.base_dim}")
    print(f"Oscillator type: {args.type} | Bandwidth: {args.bandwidth}")

    if test_only:
        save_dir = f'../saved_models/{save_path}'
        run_test(dataset_path=dataset_path, 
                 model_path=save_dir, 
                 model=model, 
                 device=device, 
                 epoch=99, 
                 test_only=test_only
                )
        exit()

    best_val_loss = float("inf")
    start_time = time.time()
    num_epochs = config.Model.epoch 
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(dataloader) * 1,
            num_training_steps=len(dataloader) * num_epochs,
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"report_{args.epoch}.txt"), "w") as report_file:
        print('Train batches:', len(dataloader))
        for epoch in range(config.Model.epoch):
            epoch_start_time = time.time()
            model.train()
            tl, ll,le, n = 0.0, 0.0, 0.0, 0
            for batch in dataloader:
                optimizer.zero_grad()
                loc_logits, time_logits, rank = model(batch)
                loc_y = batch['location_y'].view(-1).to(device)
                ts_y = batch['timeslot_y'].view(-1).to(device)
                time_loss = listmle_loss(time_logits.view(-1, 24), ts_y)
                loc_loss = listmle_loss(loc_logits, loc_y)
                seq_loss = listmle_loss(rank, loc_y)
                loss = lambda_loc * loc_loss + lambda_time * time_loss + seq_loss * lambda_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tl += time_loss.item()
                ll += loc_loss.item()
                le += seq_loss.item()
                n += 1
            average_loss = (lambda_loc * ll + lambda_time * tl + lambda_rank*le) / n
            #average_loss = (lambda_loc * ll + lambda_time * tl ) / n

            epoch_str = f"================= Epoch [{epoch + 1}/{num_epochs}] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =================\n"
            if average_loss <= best_val_loss:
                epoch_str += f"\n loc_loss: {ll/n:.4f} | time_loss: {tl/n:.4f} | rank_loss: {le/n:.4f}\n"
                epoch_str += f"Best Loss: {best_val_loss:.6f} ---> {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"
                best_val_loss = average_loss
            else:
                epoch_str += f"\n loc_loss: {ll/n:.4f} | time_loss: {tl/n:.4f} | rank_loss: {le/n:.4f}\n"
                epoch_str += f"Best Loss: {best_val_loss:.6f} | Epoch Loss: {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"

            report_file.write(epoch_str + '\n\n')
            report_file.flush()
            print(epoch_str)
            if (epoch+1) % config.Model.test_epoch == 0:
            # if (epoch+1) % 10 == 0:
                run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=epoch, test_only=test_only)

    end_time = time.time()
    total_time = end_time - start_time

    with open(os.path.join(save_dir, f"report_{args.epoch}.txt"), "a") as report_file:
        report_file.write(f"Total Running Time: {total_time:.2f} seconds\n")

    print(f"\nModel done.\n")###有1111是原本，不是全combine
