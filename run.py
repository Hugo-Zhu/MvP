import torch
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers import XLNetTokenizer, RobertaTokenizer

from alegant import Runner
from alegant import DataModuleConfig, DataModule
from src.trainers.trainer_moe import MoeTrainer
from src.modeling import MoeConfig, MoE
from src.dataset import DataModuleConfig, KaggleDataModule
from src.utils import process_data, statistical_analysis

def load_data(file_path):
    data = pickle.load(open(file_path, 'rb'))
    text = data['posts_text']
    label = data['annotations']
    processed_data = process_data(text, label)
    return processed_data
    

@ dataclass()
class KaggleConfig:
    data_path: str 
    pretrain_type: str = 'bert'
    model_dir: str = 'bert-base-cased'
    max_post: int = 50
    max_len: int = 70
    

class KaggleDataset(Dataset):
    def __init__(self, config: KaggleConfig):
        self.config = config
        self.data = load_data(self.config.data_path)

        if self.config.pretrain_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]'])
        elif self.config.pretrain_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['<pad>', '<cls>'])
        elif self.config.pretrain_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_dir)
            self.pad, self.cls = self.tokenizer.convert_tokens_to_ids(['<pad>', '<s>'])
        else:
            raise NotImplementedError

        self.convert_features() # self.data.keys(): ['posts', 'label0', 'label1', 'label2', 'label3', 'post_tokens_id']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return: 单一用户的: post_tokens_ids, label0, label1, label2, label3
        """
        e = self.data[idx]
        items = e['post_tokens_id'], e['label0'], e['label1'], e['label2'], e['label3']
        items_tensor = tuple(torch.tensor(t) for i,t in enumerate(items))
        return items_tensor
    

    def _tokenize(self, text):
        tokenized = self.tokenizer.tokenize(text)   # text --> tokenized
        ids = self.tokenizer.convert_tokens_to_ids(tokenized) # tokenized --> ids
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids) # 加CLS和SEP
        return input_ids
    
    def _pad_and_truncate(self, input_ids):
        pad_len = self.config.max_len - len(input_ids)
        if pad_len > 0:
            if self.config.pretrain_type == 'bert':
                input_ids += [self.pad] * pad_len
            elif self.config.pretrain_type == 'xlnet':
                input_ids = [input_ids[-1]] + input_ids[:-1]
                input_ids += [self.pad] * pad_len
            elif self.config.pretrain_type == 'roberta':
                input_ids += [self.pad] * pad_len
            else:
                raise NotImplementedError
        else:
            if self.config.pretrain_type == 'bert':
                input_ids = input_ids[:self.config.max_len - 1] + input_ids[-1:]
            elif self.config.pretrain_type == 'xlnet':
                input_ids = [input_ids[-1]]+ input_ids[:self.config.max_len - 2] + [input_ids[-2]]
            elif self.config.pretrain_type == 'roberta':
                input_ids = input_ids[:self.config.max_len - 1] + input_ids[-1:]
            else:
                raise NotImplementedError
        assert (len(input_ids) == self.config.max_len)
        return input_ids
        
    def convert_feature(self, i):
        post_tokens_id=[]
        for post in self.data[i]['posts'][:self.config.max_post]:
            input_ids = self._tokenize(post)
            input_ids = self._pad_and_truncate(input_ids)
            post_tokens_id.append(input_ids)

        real_post = len(post_tokens_id)
        for j in range(self.config.max_post-real_post):
            post_tokens_id.append([self.pad]*self.config.max_len)
        self.data[i]['post_tokens_id'] = post_tokens_id

    def convert_features(self):
        for i in tqdm(range(len(self.data))):
            self.convert_feature(i)


class KaggleDataModule(DataModule):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        kaggle_config = KaggleConfig(data_path=self.config.train_data_path)
        self.train_dataset = KaggleDataset(config=kaggle_config)
        kaggle_config = KaggleConfig(data_path=self.config.val_data_path)
        self.val_dataset = KaggleDataset(config=kaggle_config)
        kaggle_config = KaggleConfig(data_path=self.config.test_data_path)
        self.test_dataset = KaggleDataset(config=kaggle_config)
    
    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()



class MoeRunner(Runner):
    def setup(self):
        model = MoE(config=MoeConfig(**self.args.moe_config))
        if self.args.dataset == "kaggle":
            data_module = KaggleDataModule(DataModuleConfig(**self.args.data_module_config))
        else: # TODO
            raise NotImplementedError
        trainer_class = MoeTrainer
        return model, data_module, trainer_class



if __name__ == "__main__":
    runner = MoeRunner()
    runner.run()