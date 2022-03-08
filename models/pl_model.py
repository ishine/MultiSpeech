from models.network import Multispeech
import torch
from torch import nn
import pytorch_lightning as pl
import transformers

class PL_model(pl.LightningModule):
    def __init__(self, train_config, data_config, gpu_num):
        super().__init__()
        self.save_hyperparameters()
        self.model = Multispeech(train_config, data_config)
        self.train_config = train_config
        self.data_config = data_config
        self.gpu_num = gpu_num
        
    def forward(self, data):
        return self.model(**data)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        pred, _ = self.forward(data)
        mel_loss, bce_loss = trasnformer_loss(**labels, **pred)
        train_loss = mel_loss + bce_loss
        
        self.log("train_loss", train_loss, on_epoch=True)
        self.log("mel_loss", mel_loss, on_epoch=True)
        self.log("bce_loss", bce_loss, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        pred, _ = self.forward(data)
        mel_loss, bce_loss = trasnformer_loss(**labels, **pred)
        val_loss = mel_loss + bce_loss
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("mel_loss", mel_loss, on_epoch=True)
        self.log("bce_loss", bce_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.lr)
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config.warmup_step,
            num_training_steps=self.train_config.training_step
        )
        return [optimizer], [scheduler]
    
    
def trasnformer_loss(mel, stop_tokens, pred_mel, pred_stop_tokens, bce_weight=8):

    mel_loss = nn.L1Loss()(pred_mel, mel)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weight))(pred_stop_tokens.squeeze(), stop_tokens)

    return mel_loss, bce_loss
