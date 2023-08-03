import torch
from transformers import BertModel, BertConfig, logging

logging.set_verbosity_error()


class PeptideBERT(torch.nn.Module):
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()

        self.protbert = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd',
            config=bert_config,
            ignore_mismatched_sizes=True
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(bert_config.hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs, attention_mask):
        output = self.protbert(inputs, attention_mask=attention_mask)

        return self.head(output.pooler_output)


def create_model(config):
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['hidden_size'],
        num_hidden_layers=config['network']['hidden_layers'],
        num_attention_heads=config['network']['attn_heads'],
        hidden_dropout_prob=config['network']['dropout']
    )
    model = PeptideBERT(bert_config).to(config['device'])

    return model


def cri_opt_sch(config, model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )

    return criterion, optimizer, scheduler
