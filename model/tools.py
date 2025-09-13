import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import MyDataset

def custom_collate(batch, device, args):
    batch_dict = {
        'user': torch.tensor([item['user'] for item in batch]).to(device),
        'location_x': torch.stack([torch.tensor(item['location_x']) for item in batch]).to(device),
        'hour': torch.stack([torch.tensor(item['hour']) for item in batch]).to(device),
        'location_y': torch.tensor([item['location_y'] for item in batch]).to(device),
        'timeslot_y': torch.tensor([item['timeslot_y'] for item in batch]).to(device),
        'hour_mask': torch.stack([torch.tensor(item['hour_mask']) for item in batch]).to(device),
        'prob_matrix_time_individual': torch.stack([torch.tensor(item['prob_matrix_time_individual']) for item in batch]).to(device),
    }

    if args.topic > 0:
        batch_dict['user_topic_loc'] = torch.stack([torch.tensor(item['user_topic_loc'], dtype=torch.float32) for item in batch]).to(device)

    return batch_dict

def get_mapper(dataset_path):
    location_mapper_path = os.path.join(dataset_path, 'location_mapper.npy')
    user_mapper_path = os.path.join(dataset_path, 'user_mapper.npy')

    if os.path.exists(location_mapper_path) and os.path.exists(user_mapper_path):
        return

    location_set = set()
    user_set = set()

    with open(os.path.join(dataset_path, 'train.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()
    with open(os.path.join(dataset_path, 'test.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()

    location2id = {location: idx for idx, location in enumerate(location_set)}
    user2id = {user: idx for idx, user in enumerate(user_set)}
    print('unique location num:', len(location2id))
    print('unique user num:', len(user2id))
    yml_modified = 'y'
    if yml_modified:
        np.save(location_mapper_path, location2id)
        np.save(user_mapper_path, user2id)
    else:
        print('Program Exit')
        exit()

    
def run_test(dataset_path, model_path, model, device, epoch, test_only, args, entropy_thres=0.6, entropy_low=None, chaotic_high_list=None):
    dataset = MyDataset(args=args, dataset_path=dataset_path, device=device, load_mode='test')

    batch_size = args.batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, collate_fn=lambda batch: custom_collate(batch, device, args))
    print('Test batches:', len(dataloader))

    if test_only:
        saved_model_path = os.path.join(model_path, f'model_checkpoint_epoch{epoch}.pth')
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"Checkpoint not found: {saved_model_path}. Make sure you saved this epoch.")
        model.load_state_dict(torch.load(saved_model_path, map_location=device)['model_state_dict'])
        model.to(device)

    model.eval()
    precision_loc = 0
    top_k_values = [1, 3, 5, 10]
    top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
    total_samples = 0

    def evaluate(output, label, ks):
        topk_correct_counts = [
            torch.sum(
                (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
            ).item()
            for top_k in ks
        ]
        return np.array(topk_correct_counts)

    def calculate_mrr(output, true_labels):
        res = 0.0
        for i, pred in enumerate(output):
            sorted_indices = torch.argsort(pred, descending=True)
            true_index = np.where(true_labels[i].cpu() == sorted_indices.cpu())[0]
            if len(true_index) > 0:
                res += 1.0 / (true_index[0] + 1)
        return res

    def calc_prefix_entropy_and_return(history_locs: torch.Tensor, future_locs: torch.Tensor):
        B, L = history_locs.shape
        ent_list = []
        ret_list = []
        for b in range(B):
            row_hist = history_locs[b]
            row_y = future_locs[b]
            for t in range(L):
                if t == 0:
                    ent_list.append(0.0)
                    ret_list.append(False)
                    continue
                prefix = row_hist[:t]
                uniques, counts = torch.unique(prefix, return_counts=True)
                m = int(uniques.numel())
                if m <= 1:
                    ent_list.append(0.0)
                else:
                    p = counts.float() / float(t)
                    H = -(p * torch.log(p + 1e-12)).sum()
                    ent_list.append((H / torch.log(torch.tensor(float(m), device=prefix.device))).item())
                # ret_list.append(bool((row_y[t].item() in set(prefix.tolist()))))
                #prefix_list = prefix.tolist()
                #ret_list.append(bool((row_y[t].item() in set(prefix_list))))
                is_in_prefix = torch.any(prefix == row_y[t])
                ret_list.append(is_in_prefix.item())
        ent_norm = torch.tensor(ent_list, dtype=torch.float32, device=future_locs.device)
        is_return = torch.tensor(ret_list, dtype=torch.bool, device=future_locs.device)
        return ent_norm, is_return

    # buckets (improved)

    bucket_hits_map = {}
    bucket_mrr_map = {}
    bucket_count_map = {}
    def ensure_bucket(name):
        if name not in bucket_hits_map:
            bucket_hits_map[name] = np.zeros(len(top_k_values), dtype=np.int64)
            bucket_mrr_map[name] = 0.0
            bucket_count_map[name] = 0

    high_list = None
    low_list = None
    if chaotic_high_list is not None:
        try:
            high_list = [float(x) for x in str(chaotic_high_list).split(',') if x]
        except Exception:
            high_list = None
    if chaotic_high_list is not None:
        try:
            low_list = [float(x) for x in str(chaotic_high_list).split(',') if x]
        except Exception:
            low_list = None

    with torch.no_grad():
        for batch_data in dataloader:
            outputs = model(batch_data, args)
            location_output = outputs[0] if isinstance(outputs, tuple) else outputs
            location_y = batch_data['location_y']
            location_y = location_y.view(-1)
            B = location_y.size(0)
            total_samples += B

            top_k_correct_loc += evaluate(location_output, location_y, top_k_values)
            precision_loc += calculate_mrr(location_output, location_y)

            # entropy-based split using prefix history (no leakage)
            history_locs = batch_data['location_x']  # [B, L]
            future_locs = batch_data['location_y']  # [B, L]
            ent_norm, is_return_flat = calc_prefix_entropy_and_return(history_locs, future_locs)
            if entropy_low is not None:
                name = 'periodic'
                ensure_bucket(name)
                mask = (ent_norm <= float(entropy_low)) & is_return_flat
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    bucket_count_map[name] += idx.numel()
                    bucket_hits_map[name] += evaluate(location_output[idx], location_y.view(-1)[idx], top_k_values)
                    bucket_mrr_map[name] += calculate_mrr(location_output[idx], location_y.view(-1)[idx])
            if low_list:
                for t in low_list:
                    name = f'periodic@{t}'
                    ensure_bucket(name)
                    mask = (ent_norm < t) & is_return_flat
                    if mask.any():
                        idx = mask.nonzero(as_tuple=True)[0]
                        bucket_count_map[name] += idx.numel()
                        bucket_hits_map[name] += evaluate(location_output[idx], location_y.view(-1)[idx], top_k_values)
                        bucket_mrr_map[name] += calculate_mrr(location_output[idx], location_y.view(-1)[idx])

            if high_list:
                for t in high_list:
                    name = f'chaotic@{t}'
                    ensure_bucket(name)
                    mask = (ent_norm >= t)
                    if mask.any():
                        idx = mask.nonzero(as_tuple=True)[0]
                        bucket_count_map[name] += idx.numel()
                        bucket_hits_map[name] += evaluate(location_output[idx], location_y.view(-1)[idx], top_k_values)
                        bucket_mrr_map[name] += calculate_mrr(location_output[idx], location_y.view(-1)[idx])

    top_k_accuracy_loc = [count / total_samples * 100 for count in list(top_k_correct_loc)]
    reported_epoch = epoch
    mode_tag = 'testonly' if test_only else 'train'
    result_str = "*********************** Test ***********************\n"
    result_str += f"base_dim: {args.dim} | dim: {args.dim}\n"
    result_str += f"AT_type: {args.at} | topic_num: {args.topic}\n"
    result_str += f"encoder: {args.encoder}\n"
    result_str += f"Mode: {mode_tag} | Epoch {reported_epoch}: Total {total_samples} predictions on Next Location:\n"
    for k, accuracy in zip(top_k_values, top_k_accuracy_loc):
        result_str += f"Acc@{k}: {accuracy:.2f}\n"
    result_str += f"MRR: {precision_loc * 100 / total_samples:.2f}"
    result_save = top_k_accuracy_loc
    result_save.append(precision_loc * 100 / total_samples)
    result_save = np.array(result_save)
    np.save(
        f"{model_path}/acc_{mode_tag}_{reported_epoch}_topic{args.topic}"
        f"_at{args.at}"
        f"_dim{args.dim}_encoder{args.encoder}"
        f"_seed{args.seed}.npy",
        result_save)

    print(result_str)

    with open(os.path.join(model_path, 'results.txt'), 'a', encoding='utf8') as res_file:
        res_file.write(result_str + '\n\n')
    if high_list:
        for t in high_list:
            name = f'chaotic@{t}'
            if name in bucket_hits_map:
                cnt = max(bucket_count_map[name], 1)
                accs = (bucket_hits_map[name] / cnt) * 100
                mrr_val = (bucket_mrr_map[name] / cnt) * 100
                s = (f"[{name.upper()}] count={cnt} | "
                     + " ".join([f"Acc@{k}: {acc:.2f}" for k, acc in zip(top_k_values, accs)])
                     + f" | MRR: {mrr_val:.2f}")
                print(s)


def train_epoch(model, dataloader, optimizer, loss_fn, scheduler):
    model.train()
    total_loss_epoch = 0.0

    for batch_data in dataloader:
        location_output = model(batch_data)
        location_y = batch_data['location_y'].view(-1)
        location_loss = loss_fn(location_output, location_y)
        total_loss = location_loss.sum()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        total_loss_epoch += total_loss.item()

    return total_loss_epoch / len(dataloader)


def save_checkpoint(save_dir, model, optimizer, best_val_loss, epoch):
    save_path = os.path.join(save_dir, f"model_checkpoint_epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, save_path)


def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


