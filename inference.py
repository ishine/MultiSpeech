# %%
from configs import DataConfig, TrainConfig
import os
from models.pl_model import PL_model
import pytorch_lightning as pl
from dataset import *
import torch
import matplotlib.pyplot as plt
# %%

out_path = './experiments/using_phonemes'
data_config = DataConfig(
    root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
    train_csv='metadata_train.csv',
    val_csv='metadata_val.csv'
)
train_config = TrainConfig(
    batch_size=64,
    training_step=16000,
    warmup_step=16000*0.1,
    checkpoint_path= os.path.join(out_path, 'checkpoint'),
    log_dir= os.path.join(out_path, 'tensorboard')
)

model= PL_model(train_config, data_config, gpu_num=1).load_from_checkpoint('./experiments/using_phonemes/checkpoint/epoch=170-val_loss=0.73191.ckpt')

# %%
model.eval()
model.freeze()
# %%
data = PartitionPerEpochDataModule(4, data_config, num_workers=8)
data.setup()
# %%
for i in data.val_dataloader():
    input_, label = i
    break
# %%
out, attn = model.forward(input_)
# %%
out.keys()
# %%
plt.figure(figsize=(12, 12))
plt.imshow(attn['enc_dec_attn_list'][0].transpose(1, 2)[3])
# %%
attn['enc_dec_attn_list'][0].size()
# %%
torch.argmax(out['pred_stop_tokens'])
# %%
label['stop_tokens']
# %%
label['stop_tokens']
# %%
out['pred_stop_tokens'].size()
# %%
plt.imshow(attn['enc_attn_list'][3].transpose(1, 2)[0])
# %%
attn.keys()
# %%
plt.imshow(torch.triu(torch.ones(80, 80), diagonal=1).bool())
# %%
torch.triu(torch.ones(80, 80), diagonal=1).bool()
# %%
input_['pos_mel'][0]
# %%
