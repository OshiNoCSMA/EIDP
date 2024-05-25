import torch
import random
from torch.utils.data import Dataset

def neg_sample(seq_item, itemset):
    seq_item_set = set(seq_item)
    neg_item_idx = random.randint(0, len(itemset)-1)
    neg_item = itemset[neg_item_idx]
    while neg_item in seq_item_set:
        neg_item_idx = random.randint(0, len(itemset)-1)
        neg_item = itemset[neg_item_idx]
    return neg_item

class BehaviorSetSequentialRecDataset(Dataset):
    def __init__(self, config, data, data_type="train"):
        self.config = config
        (self.train_data,
         self.valid_data,
         self.bvt_data,
         self.test_data,
         userset,
         itemset,
         self.all_behavior_type) = data

        self.itemset = list(itemset)
        self.userset = list(userset)
        self.data_type = data_type
        self.maxlen = config['maxlen']

    def _pack_up_data_to_tensor(self, user_id, term_seq, lack=False):
        seq_item = [term[0] for term in term_seq]
        seq_bs = [term[1] for term in term_seq]

        input_item, input_bs = seq_item[:-1], seq_bs[:-1]
        target_item, target_bs = seq_item[1:], seq_bs[1:]
        if lack:
            ground_truth = [0]
        else:
            ground_truth = [seq_item[-1]]

        target_neg = []
        for _ in target_item:
            target_neg.append(neg_sample(seq_item, self.itemset))

        pad_len = self.maxlen - len(input_item)
        input_item = [0] * pad_len + input_item
        input_bs = [[0] * self.all_behavior_type] * pad_len + input_bs

        target_item = [0] * pad_len + target_item
        target_bs = [[0] * self.all_behavior_type] * pad_len + target_bs

        target_neg = [0] * pad_len + target_neg

        input_item = input_item[-self.maxlen:]
        target_item = target_item[-self.maxlen:]
        target_neg = target_neg[-self.maxlen:]
        input_bs = input_bs[-self.maxlen:]
        target_bs = target_bs[-self.maxlen:]

        # warning msg part
        assert len(input_item) == self.maxlen
        assert len(target_item) == self.maxlen
        assert len(target_neg) == self.maxlen

        one_id_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_item, dtype=torch.long),
            torch.tensor(input_bs, dtype=torch.float),
            torch.tensor(target_item, dtype=torch.long),
            torch.tensor(target_bs, dtype=torch.float),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(ground_truth, dtype=torch.long)
        )
        return one_id_tensors

    def __getitem__(self, index):
        user_id = index + 1
        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            train_term_list = self.train_data[user_id]
            one_id_tensors = self._pack_up_data_to_tensor(user_id, train_term_list, lack=False)

        elif self.data_type == "valid":
            valid_term_list = self.train_data[user_id] + self.valid_data[user_id]
            lack = len(self.valid_data[user_id]) == 0
            one_id_tensors = self._pack_up_data_to_tensor(user_id, valid_term_list, lack=lack)

        else:
            test_term_list = self.train_data[user_id] + self.valid_data[user_id] + \
                              self.bvt_data[user_id] + self.test_data[user_id]
            lack = len(self.test_data[user_id]) == 0
            one_id_tensors = self._pack_up_data_to_tensor(user_id, test_term_list, lack=lack)

        return one_id_tensors

    def __len__(self):
        return len(self.train_data)