import argparse,os,time,torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch.nn as nn
from dataloader import MyDataset
from cnolp import CNOLP_tc, CNOLP_mp
import torch.nn.functional as F
from tools import run_test, train_epoch, get_mapper, custom_collate

def get_dataset_info(dataset_path):
    user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
    location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()
    return len(location2id), len(user2id)
parser = argparse.ArgumentParser()
# ==================== Exp settings ====================
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--test', type=bool, default=False)

# ==================== Dataset ====================
parser.add_argument('--dataset', type=str, default='TC',help='dataset name')
parser.add_argument('--topic', type=int, default=350, help='LDA topic num')
parser.add_argument('--sequence_length', type=int, default=20, help='sequence length for data processing')

# ==================== Model ====================
parser.add_argument('--dim', type=int, default=16, 
            help='must be a multiple of 4 //   TC dataset --> 16  && MP dataset --> 8')
parser.add_argument('--at', type=str, default='osc', 
            help='Chaotic Neural Oscillator Attention')
parser.add_argument('--encoder', type=str, default='trans', 
            help='encoder type')
parser.add_argument('--type', type=str, default='cnoa_tc', 
            help='parameter in Chaotic Neural Oscillator Attention')
parser.add_argument('--bandwidth', type=float, default=0.5, 
            help='bandwidth parameter for gaussian kernel --> 0.5 / 1.0 / 1.5 / 2.0')

# ==================== Training ====================
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--epoch', type=int, default=100, help='epoch num')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizer')
parser.add_argument('--entropy_thres', type=float, default=0.8, help='(legacy) single threshold for chaotic split; kept for back-compat')
parser.add_argument('--test_epoch', type=int, default=100, help='run evaluation every N epochs (overrides settings.yml)')
parser.add_argument('--entropy_low', type=float, default=0.4, help='periodic bucket: H_norm <= entropy_low AND return step /// 0.35 or 0.40')
parser.add_argument('--chaotic_high_list', type=str, default='0.75,0.8,0.85,0.9', help='comma-separated highs for chaotic: H_norm >= t OR exploration step')
args = parser.parse_args()
gpu_list = args.gpu
torch.manual_seed(args.seed)
dataset_path = f'./data/{args.dataset}'
timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
save_dir = os.path.join("saved_models", args.dataset)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_only = args.test
if __name__ == '__main__':
    get_mapper(dataset_path=dataset_path)
    num_locations, num_users = get_dataset_info(dataset_path)

    dataset = MyDataset(args=args, dataset_path=dataset_path, device=device, load_mode='train')
    batch_size = args.batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                            collate_fn=lambda batch: custom_collate(batch, device, args))
    if args.dataset == "TC":
        model = CNOLP_tc(args, num_locations, num_users, args.topic)
    else: 
        model = CNOLP_mp(args, num_locations, num_users, args.topic)
    model.to(device)
    ce_time = torch.nn.CrossEntropyLoss(reduction='mean')
    lambda_time = 0.4
    lambda_loc = 0.9
    lambda_rank = 0.6
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * batch_size} | Total trainable parameters: {total_params}")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.999), eps=1e-08, 
        weight_decay=args.weight_decay, amsgrad=False,
        maximize=False, foreach=None, capturable=False, 
        differentiable=False, fused=None
    )

    print(f"Dataset: {args.dataset} | Device: {device} | Model: {args.encoder}")
    print(f"AT type: {args.at} | topic_num: {args.topic} | dim: {args.dim}")

    if test_only:
        save_dir = os.path.join("saved_models", args.dataset)
        run_test(dataset_path=dataset_path, 
                 model_path=save_dir, 
                 model=model, 
                 device=device, 
                 epoch=args.epoch, 
                 test_only=test_only,
                 args=args,
                 entropy_thres=args.entropy_thres,
                 entropy_low=args.entropy_low,
                 chaotic_high_list=args.chaotic_high_list
                )
        exit()

    start_time = time.time()
    num_epochs = args.epoch 
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(dataloader) * 1,
            num_training_steps=len(dataloader) * num_epochs,
    )

    os.makedirs(save_dir, exist_ok=True)
    now_time = time.time()
    with open(os.path.join(save_dir, f"report_{args.epoch}_{now_time}.txt"), "w") as report_file:
        print('Train batches:', len(dataloader))
        for epoch in range(1, args.epoch + 1):
            epoch_start_time = time.time()
            model.train()
            tl, ll,le, n = 0.0, 0.0, 0.0, 0
            for batch in dataloader:
                optimizer.zero_grad()
                loc_logits, time_logits, rank = model(batch, args)
                loc_y = batch['location_y'].view(-1).to(device)
                ts_y = batch['timeslot_y'].view(-1).to(device)
                time_loss = F.cross_entropy(time_logits.view(-1, 24), ts_y, reduction='mean')
                loc_loss = F.cross_entropy(loc_logits, loc_y, reduction='mean')
                seq_loss = F.cross_entropy(rank, loc_y, reduction='mean')
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

            epoch_str = f"================= Epoch [{epoch}/{num_epochs}] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} =================\n"
            epoch_str += f"\n loc_loss: {ll/n:.4f} | time_loss: {tl/n:.4f} | rank_loss: {le/n:.4f}\n"
            epoch_str += f"Epoch Loss: {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"

            report_file.write(epoch_str + '\n\n')
            report_file.flush()
            print(epoch_str)
            if epoch % args.test_epoch == 0:
                checkpoint_epoch = epoch
                torch.save({
                    'epoch': checkpoint_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, f'model_checkpoint_epoch{checkpoint_epoch}.pth'))
                # Call evaluation with the same human-readable epoch number
                run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=checkpoint_epoch, test_only=False, args=args, entropy_thres=args.entropy_thres, entropy_low=args.entropy_low, chaotic_high_list=args.chaotic_high_list)

    end_time = time.time()
    total_time = end_time - start_time

    final_checkpoint = {
        'epoch': args.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(save_dir, f'model_checkpoint_epoch{args.epoch}.pth'))
    print(f"Final model saved at epoch {args.epoch}")

    with open(os.path.join(save_dir, f"report_{args.epoch}_{now_time}.txt"), "a") as report_file:
        report_file.write(f"Total Running Time: {total_time:.2f} seconds\n")

    print(f"\nModel done.\n")
