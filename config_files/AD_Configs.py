class Config(object):
    def __init__(self,):
        self.RANDOM_SEED = 42

        # data loading para specific, do not change
        self.S_OVERLAPPING = 0.5
        self.S_TIMESTAMPS = 256

        # pretrain configs
        self.dataset = 'AD'
        self.input_dims = 16
        self.output_dims = 320
        self.pretrain_lr = 1e-4
        self.depth = 12
        self.pretrain_batch_size = 128
        self.trial_shuffle = True  # do trial or batch shuffle
        self.verbose = True
        self.n_epochs = 100
        self.masks = ['all_true', 'all_true', 'continuous', 'continuous']  # patient, trial, sample, observation
        self.factors = [1.0, 1.0, 1.0, 1.0]  # patient, trial, sample, observation

        # finetune configs
        self.num_classes = 2

        # 100% label
        self.fraction_100 = 1.0
        self.finetune_batch_size_100 = 128
        self.finetune_epochs_100 = 50
        self.finetune_lr_100 = 1e-4

        # 30% label
        self.fraction_30 = 0.3
        self.finetune_batch_size_30 = 128
        self.finetune_epochs_30 = 50
        self.finetune_lr_30 = 1e-4

        # 10% label
        self.fraction_10 = 0.1
        self.finetune_batch_size_10 = 128
        self.finetune_epochs_10 = 100
        self.finetune_lr_10 = 1e-4

        # 1% label
        self.fraction_1 = 0.01
        self.finetune_batch_size_1 = 128
        self.finetune_epochs_1 = 100
        self.finetune_lr_1 = 1e-4


