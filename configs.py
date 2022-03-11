from dataclasses import dataclass

@dataclass
class DataConfig():
    """
    Data Settings
    """
    
    # Audio Config
    n_fft: int = 2048
    sr: int = 22050
    preemphasis: float = 0.97
    frame_shift: float = 0.0125  # seconds
    frame_length: float = 0.05  # seconds
    hop_length: int = 256       #(sr*frame_shift)
    win_length: int = 1024      #(sr*frame_length)
    n_mels: int  = 80  # Number of Mel banks to generate
    power: float = 1.2  # Exponent for amplifying the predicted magnitude
    min_level_db: int = -100
    ref_level_db: int = 20
    max_db: int = 100
    ref_db: int = 20
    n_iter: int = 60
    # Text Config 
    # previous setting(using text)
    # cleaners: str = 'english_cleaners'
    # symbol_length: int = 149
    # Modeified Setting(using phoneme)
    cleaners: str = "phoneme_cleaners"
    use_phonemes: bool =True
    language: str ="en-us"
    phoneme_cache_path: str= './phonemes/'
    symbol_length: int = 130
    enable_eos_bos: bool = True
    # Data path config
    train_csv: str = 'metadata.csv'
    val_csv: str = 'metadata.csv'
    root_dir: str = './data/LJSpeech-1.1'
    train_samples: int = 12000
    
@dataclass
class TrainConfig():
    """
    Train Setting
    """
    bce_weight: int = 8
    hidden_size: int = 256
    decoder_prenet_hidden_size: int = 32
    n_head: int = 2
    embedding_size: int = 256
    n_layers: int = 4
    dropout_p: int = 0.2
    warmup_step: int = 1600
    training_step: int = 16000
    lr: float = 0.001
    batch_size: int = 128
    checkpoint_path: str = './models/checkpoint/'
    log_dir: str = './models/tensorboard/'
