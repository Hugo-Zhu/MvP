import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from loguru import logger
from ..loss import ce_loss
from ..utils import compute_metrics_all_class
from alegant.trainer import TrainingArguments, Trainer, DataModule, autocast


class MoeTrainer(Trainer):
    def __init__(self,
        args: TrainingArguments, 
        model: nn.Module, 
        data_module: DataModule):       
        super().__init__(args, model, data_module)

        self.criterion = ce_loss()


    def training_step(self, batch, batch_idx):
        post_tokens_ids = batch[0].to(self.args.device)  # (B, N, L)
        label_list = [label.to(self.args.device) for label in batch[1:]]      # label_list: num_traits个(B,)
        with autocast(enabled=self.args.amp):
            outputs = self.model(post_tokens_ids)
            logits_list = outputs.get("logits_list")
            loss_cl = torch.sum(outputs.get("loss_cl"))
            loss_det = self.criterion(logits_list, label_list)
            loss = loss_det + loss_cl
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        post_tokens_ids = batch[0].to(self.args.device)  # (B, N, L)
        label_list = [label.to(self.args.device) for label in batch[1:]]      # label_list: num_traits个(B,)
        with autocast(enabled=self.args.amp):
            outputs = self.model(post_tokens_ids)
            logits_list = outputs.get("logits_list")
            loss_cl = torch.sum(outputs.get("loss_cl"))
            loss_det = self.criterion(logits_list, label_list)
            loss = loss_det + loss_cl
        return {"loss": loss, "logits_list": logits_list, "label_list": label_list}


    def validation_epoch_end(self, outputs):
        logits_all_batch = []
        labels_all_batch = []
        for item in outputs:
            logits_all_batch.append(torch.stack(item["logits_list"]).detach().cpu())
            labels_all_batch.append(torch.stack(item["label_list"]).detach().cpu())
        
        logits_all_sample = torch.cat(logits_all_batch, dim=-2).float()
        preds_all_sample = torch.argmax(F.softmax(logits_all_sample, dim=-1), dim=-1)
        labels_all_sample = torch.cat(labels_all_batch, dim=-1)
        preds_all_class = preds_all_sample
        labels_all_class = labels_all_sample
        metrics = compute_metrics_all_class(preds_all_class, labels_all_class, average="macro")
        avg_f1 = metrics.get("avg_f1")

        self.logger.add_scalar("avg_f1", avg_f1, self.num_eval)
        
        # Update best metrix
        if avg_f1 > self.best_f1:
            self.best_f1 = avg_f1
            if self.args.do_save:
                self.save_checkpoint()
        logger.info(f"avg_f1: {avg_f1:.5f}; best_f1: {self.best_f1:.5f}")


    def test_epoch_end(self, outputs):
        logits_all_batch = []
        labels_all_batch = []
        for item in outputs:
            logits_all_batch.append(torch.stack(item["logits_list"]).detach().cpu())
            labels_all_batch.append(torch.stack(item["label_list"]).detach().cpu())
        
        logits_all_sample = torch.cat(logits_all_batch, dim=-2).float()
        preds_all_sample = torch.argmax(F.softmax(logits_all_sample, dim=-1), dim=-1)
        labels_all_sample = torch.cat(labels_all_batch, dim=-1)

        preds_all_class = preds_all_sample
        labels_all_class = labels_all_sample

        metrics = compute_metrics_all_class(preds_all_class, labels_all_class, average="macro")
        avg_f1 = metrics.get("avg_f1")

        logger.success(f"TESTING RESULTS: {metrics}")


    def configure_optimizers(self, num_training_steps):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # Parameter groups: 1. parms of PLM, 2.params of other components
        params_plm = []
        no_decay_params_plm = []
        params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad == False:
                continue

            elif "bert" in name:
                if any(nd in name for nd in no_decay):
                    no_decay_params_plm.append(param)
                else:
                    params_plm.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    params.append(param)

        # Grouped parameters
        optimizer_grouped_parameters = [
            {
                "params": params_plm,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate_plm
            },
            {   
                "params": no_decay_params_plm,
                "weight_decay": 0.0,
                "lr": self.args.learning_rate_plm
            },
            {
                "params": params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            }
        ]

        # Define the optimizer
        optimizer = Adam(optimizer_grouped_parameters)
        scheduler = None

        return optimizer, scheduler
