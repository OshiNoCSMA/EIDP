import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.sparse import csr_matrix
from torch.utils.tensorboard import SummaryWriter

from utils import *

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config):
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.behavior_types = config['behavior_types']
        self.hidden_dims = config['hidden_dims']

        self.config = config
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        self.output_path = model.save_path

        self.max_epochs = config['max_epochs']
        self.log_freq = config['log_freq']
        self.eval_freq = config['eval_freq']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        # Loss Function
        self.tau = config['tau']

        betas = (config['adam_beta1'], config['adam_beta2'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=betas, weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    factor=config['decay_factor'],
                                                                    verbose=True,
                                                                    min_lr=config['min_lr'],
                                                                    patience=config['patience'])

        self.tensorboard_on = config['tensorboard_on']
        self.writer = None
        if self.tensorboard_on:
            self.writer = SummaryWriter(os.path.join(config['run_dir'],
                                                     f"{self.model.model_name}_{self.model.no}"))
            self._create_model_training_folder(self.writer)
        self.eval_item_mask = self.generate_eval_item_mask()
        self.test_item_mask = self.generate_test_item_mask()
        self.print_out_epoch = 0

    def _create_model_training_folder(self, writer):
        model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)

    def generate_eval_item_mask(self):
        row = []
        col = []
        data = []
        for _, id_tensors in enumerate(self.valid_dataloader):
            uid, input_item = id_tensors[0], id_tensors[1]
            uid = uid.numpy()
            input_item = input_item.numpy()
            for idx, u in enumerate(uid):
                # for padding idx 0
                row.append(u)
                col.append(0)
                data.append(1)
                for i in input_item[idx]:
                    row.append(u)
                    col.append(i)
                    data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        eval_item_mask = csr_matrix((data, (row, col)),
                                    shape=(self.user_num + 1, self.item_num + 1))

        return eval_item_mask

    def generate_test_item_mask(self):
        row = []
        col = []
        data = []
        for _, id_tensors in enumerate(self.test_dataloader):
            uid, input_item = id_tensors[0], id_tensors[1]
            uid = uid.numpy()
            input_item = input_item.numpy()
            for idx, u in enumerate(uid):
                # for padding idx 0
                row.append(u)
                col.append(0)
                data.append(1)
                for i in input_item[idx]:
                    row.append(u)
                    col.append(i)
                    data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        test_item_mask = csr_matrix((data, (row, col)),
                                    shape=(self.user_num + 1, self.item_num + 1))

        return test_item_mask

    def train(self):
        self.fit(self.train_dataloader)

    def valid(self):
        return self.fit(self.valid_dataloader, mode="eval")

    def test(self, record=True):
        return self.fit(self.test_dataloader, mode="test", record=record)

    def fit(self, dataloader, mode="train", record=True):
        raise NotImplementedError

    def load(self):
        self.model.load_state_dict(torch.load(self.output_path))

    def save(self):
        torch.save(self.model.cpu().state_dict(), self.output_path)
        self.model.to(self.device)

    def BCELoss(self, u, seq_output, pos_ids, neg_ids):
        '''
        Binary Cross Entropy Loss
        '''
        # (b, L, d)
        pos_emb = self.model.item_emb(pos_ids)
        neg_emb = self.model.item_emb(neg_ids)

        # (bL, d)
        D = pos_emb.size(2)
        pos_item_emb = pos_emb.view(-1, D)
        neg_item_emb = neg_emb.view(-1, D)

        # (bL, d)
        it = seq_output.contiguous().view(-1, D)

        # (bL, )
        pos_item_logits = torch.sum(pos_item_emb * it, -1)
        neg_item_logits = torch.sum(neg_item_emb * it, -1)

        # (bL, d)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.maxlen).float()

        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_item_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_item_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def calculate_all_item_prob(self, u, output):
        # (|I|+1, d)
        item_emb_weight = self.model.item_emb.weight
        # item side prob
        # (b, d) * (d, |I|+1) --> (b, |I|+1)
        item_prob = torch.matmul(output, item_emb_weight.transpose(0, 1))
        return item_prob

    def calculate_eval_metrics(self, mode, print_out_epoch, pred_item_list, ground_truth_list, record=True):
        NDCG_n_list, HR_n_list = [], []
        for k in [5, 10, 20]:
            NDCG_n_list.append(ndcg_k(pred_item_list, ground_truth_list, k))
            HR_n_list.append(hr_k(pred_item_list, ground_truth_list, k))

        eval_metrics_info = {
            "Epoch": print_out_epoch,
            "HR@5": "{:.4f}".format(HR_n_list[0]),
            "NDCG@5": "{:.4f}".format(NDCG_n_list[0]),
            "HR@10": "{:.4f}".format(HR_n_list[1]),
            "NDCG@10": "{:.4f}".format(NDCG_n_list[1]),
            "HR@20": "{:.4f}".format(HR_n_list[2]),
            "NDCG@20": "{:.4f}".format(NDCG_n_list[2]),
        }

        if self.writer is not None and record:
            if mode == 'eval':
                self.writer.add_scalars('NDCG@10', {'Valid': NDCG_n_list[1]}, print_out_epoch)
                self.writer.add_scalars('HR@10', {'Valid': HR_n_list[1]}, print_out_epoch)
            else:
                self.writer.add_scalars('NDCG@10', {'Test': NDCG_n_list[1]}, print_out_epoch)
                self.writer.add_scalars('HR@10', {'Test': HR_n_list[1]}, print_out_epoch)

        if mode == 'eval':
            print(set_color(str(eval_metrics_info), "cyan"))
        return [HR_n_list[0], NDCG_n_list[0], HR_n_list[1],
                NDCG_n_list[1], HR_n_list[2], NDCG_n_list[2]], str(eval_metrics_info)


class EIDPTrainer(Trainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config):
        super(EIDPTrainer, self).__init__(
            model, train_dataloader, valid_dataloader, test_dataloader, config
        )

    def fit(self, dataloader, mode="train", record=True):
        assert mode in {"train", "eval", "test"}
        print(set_color("Rec Model mode: " + mode, "green"))

        if mode == "train":
            self.model.train()

            early_stopping = EarlyStopping(save_path=self.output_path)
            print(set_color(f"Rec dataset Num of batch: {len(dataloader)}", "white"))

            for epoch in range(self.max_epochs):

                bce_total_loss = 0.0
                joint_total_loss = 0.0

                iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
                for i, id_tensors in iter_data:
                    '''
                    user_id: (b, 1)
                    input_item: (b, L)
                    input_bs: (b, L, bt)
                    target_item: (b, L)
                    target_bs: (b, L, bt)
                    target_neg: (b, L)
                    target_neg_bs: (b, L, bt)
                    ground_truth: (b, 1)
                    '''

                    id_tensors = tuple(t.to(self.device) for t in id_tensors)
                    (uid, input_item, input_bs,
                     target_item, target_bs, target_neg, ground_truth) = id_tensors

                    # seq_output = self.model(input_item) # SASRec original
                    seq_output = self.model((uid, input_item, input_bs, target_bs)) # (ut, it)
                    bce_loss = self.BCELoss(uid, seq_output, target_item, target_neg)

                    joint_loss = bce_loss
                    self.optimizer.zero_grad()
                    joint_loss.backward()
                    self.optimizer.step()

                    bce_total_loss += bce_loss.item()
                    joint_total_loss += joint_loss.item()

                bce_avg_loss = bce_total_loss / len(iter_data)
                joint_avg_loss = joint_total_loss / len(iter_data)
                self.scheduler.step(joint_avg_loss)

                if self.writer is not None:
                    self.writer.add_scalar('BCE Loss', bce_avg_loss, epoch)
                    self.writer.add_scalar('Joint Loss', joint_avg_loss, epoch)

                loss_info = {
                    "Epoch": epoch + 1,
                    "BCE Loss": "{:.6f}".format(bce_avg_loss),
                    "Joint Loss": "{:.6f}".format(joint_avg_loss)
                }

                if (epoch+1) % self.log_freq == 0:
                    print(set_color(str(loss_info), "yellow"))

                if (epoch+1) % self.eval_freq == 0:
                    self.print_out_epoch = epoch + 1
                    scores, _ = self.valid()
                    # _, _ = self.test()
                    early_stopping(scores, epoch+1, self.model)
                    if early_stopping.early_stop:
                        print("Early Stopping")

                        best_scores_info = {
                            "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                            "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                            "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                            "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                            "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                            "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                        }

                        print(set_color(f'\nBest Valid (' +
                                        str(early_stopping.best_valid_epoch) +
                                        ') Scores: ' +
                                        str(best_scores_info) + '\n', 'cyan'))
                        break

                    self.model.train()

            if not early_stopping.early_stop:
                print("Reach the max number of epochs!")
                best_scores_info = {
                    "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                    "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                    "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                    "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                    "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                    "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                }

                print(set_color(f'\nBest Valid (' +
                                str(early_stopping.best_valid_epoch) +
                                ') Scores: ' +
                                str(best_scores_info) + '\n', 'cyan'))

            # test phase
            self.model.load_state_dict(torch.load(self.output_path))
            _, test_info = self.test(record=False)
            print(set_color(f'\nFinal Test Metrics: ' +
                            test_info + '\n', 'pink'))

        else:
            item_mask = self.eval_item_mask
            if mode == "test":
                item_mask = self.test_item_mask

            self.model.eval()
            iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
            pred_item_list = None
            ground_truth_list = None

            with torch.no_grad():
                for i, id_tensors in iter_data:
                    id_tensors = tuple(t.to(self.device) for t in id_tensors)
                    (uid, input_item, input_bs,
                     target_item, target_bs, target_neg, ground_truth) = id_tensors

                    # seq_output = self.model(input_item)
                    seq_output = self.model((uid, input_item, input_bs, target_bs))
                    it_output = seq_output[:, -1, :]

                    # batch of recommendation results
                    # (b, |I|+1)
                    item_prob = self.calculate_all_item_prob(uid, it_output).cpu().data.numpy().copy()
                    batch_user_idx = uid.cpu().numpy()
                    item_prob[item_mask[batch_user_idx].toarray() > 0] = -np.inf # (b, |I|+1)

                    # extract top-20 prob of item idx
                    top_idx = np.argpartition(item_prob, -20)[:, -20:]
                    topn_prob = item_prob[np.arange(len(top_idx))[:, None], top_idx] # (b, 20)
                    # from large to small prob
                    topn_idx = np.argsort(topn_prob)[:, ::-1] # (b, 20)
                    batch_pred_item_list = top_idx[np.arange(len(top_idx))[:, None], topn_idx] # (b, 20)

                    if i == 0:
                        pred_item_list = batch_pred_item_list
                        ground_truth_list = ground_truth.cpu().data.numpy()
                    else:
                        pred_item_list = np.append(pred_item_list, batch_pred_item_list, axis=0)
                        ground_truth_list = np.append(ground_truth_list, ground_truth.cpu().data.numpy(), axis=0)

            return self.calculate_eval_metrics(mode, self.print_out_epoch,
                                               pred_item_list, ground_truth_list, record=record)

        if self.writer is not None:
            self.writer.close()
