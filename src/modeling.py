import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import BertModel
from dataclasses import dataclass, field


@dataclass
class MoeConfig:
    task: str = "mbti"
    use_moe: bool = True
    num_experts: int = 4
    trade_off_param: float = 0.1
    dropout_prob: float = 0.0
    temp_cl: float = 0.1
    dropout_prob_cl: float = 0.1
    pretrained_model_name_or_path: str = "bert-base-cased"


class MoE(nn.Module):
    def __init__(self, 
                config: MoeConfig ):
        super().__init__()
        self.config = config
        self.ucr = UCR(temp=self.config.temp_cl,dropout_prob=self.config.dropout_prob_cl)

        self.bert = BertModel.from_pretrained(self.config.pretrained_model_name_or_path, output_hidden_states=True)
        self.moe_adaptor = MoEAdaptorLayer(n_exps=self.config.num_experts, layers=[768, 768])
        self.pooler = AveragePooler()
        self.activation = nn.ReLU()
        num_traits = 8 if self.config.task == "mbti" else 10
        self.fc = nn.Linear(self.bert.config.hidden_size, num_traits)
        self.dropout = nn.Dropout(self.config.dropout_prob)

                
    def forward(self, post_tokens_ids):
        pad_id = 0
        batch_size, num_subsequence, max_len = post_tokens_ids.size()
        
        attention_mask = (post_tokens_ids != pad_id).float()    # (B, N, L)
        post_mask = (attention_mask.sum(-1) > 0).float()        # (B, N)
        input_ids = post_tokens_ids.view(-1, max_len)           # (B*N, L)
        attention_mask = attention_mask.view(-1, max_len)       # (B*N, L)

        embeded = self.bert(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1]
        embeded_cls = embeded[:, 0, :].reshape([batch_size, num_subsequence, -1])  # (B, N, d)
        
        if self.config.use_moe:
            out = self.moe_adaptor(embeded_cls)
        else:
            out = embeded_cls

        pooled_out = self.pooler(out, post_mask)                    # (B, d)
        pooled_out = self.dropout(pooled_out)
        logits_list = self.fc(pooled_out)
        
        batch_size, _ = logits_list.size()
        logits_list = logits_list.view(batch_size, -1, 2).transpose(0,1)  # (real_num_traits, B, 2)
        loss_cl = self.contrastive_loss(pooled_out) * self.config.trade_off_param
        
        return {"logits_list":[logits for logits in logits_list], "loss_cl":loss_cl}


class AveragePooler(nn.Module):
    def __init__(self):
        super(AveragePooler, self).__init__()
    
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).expand_as(x).float()
        masked_x = x * mask

        pooling_output = masked_x.sum(dim=1) / (mask+1e-8).sum(dim=1)
        return pooling_output


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor

    Example:
        x = torch.randn([32, 128])
        model = MoEAdaptorLayer(n_exps=3, layers=[128, 768])
        y = model(x)
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class ce_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_all_class, targets_all_class):
        """
        Calculates the cross-entropy loss for all labels and all samples.

        Args:
            logits_all_class (list): List of logits for each class.
                Each logits item contains logits for a single label.
            targets_all_class (list): List of target labels for each class.

        Returns:
            torch.Tensor: Average cross-entropy loss for all labels.
        """
        assert len(logits_all_class) == len(targets_all_class)

        num_samples = 0
        loss_all_class = []
        for i, logits in enumerate(logits_all_class):
            num_samples += logits.size(0)
            # loss = nn.functional.cross_entropy(logits, targets_all_class[i])
            loss = nn.functional.cross_entropy(logits_all_class[i], targets_all_class[i])
            loss_all_class.append(loss)
        return sum(loss_all_class) / num_samples


class UCR(nn.Module):
    def __init__(self, temp, dropout_prob) -> None:
        super().__init__()
        self.temp = temp
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs): # shape (N, hidden_dim)
        inputs_enhanced = self.dropout(inputs)
        similarity = F.cosine_similarity(inputs.unsqueeze(1), inputs_enhanced.unsqueeze(0), dim=-1)
        sim_tau = similarity/self.temp

        loss = torch.exp(sim_tau.diag()) / torch.sum(torch.exp(sim_tau), dim=1)
        return torch.mean(-torch.log(loss))
