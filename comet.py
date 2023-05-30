import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder, ProjectionHead
from models.losses import contrastive_loss
from models.losses import sample_contrastive_loss, observation_contrastive_loss, patient_contrastive_loss, trial_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from utils import split_data_label
import math
import copy
import sklearn
from sklearn.preprocessing import LabelBinarizer


class COMET:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        trial_batch_size=15,
        sample_batch_size=128,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (float): The learning rate.
            trial_batch_size (int): The batch size of trials. It used in fit.
            sample_batch_size (int): The batch size of samples(split/segmented data). It used in linear evaluation and fine-tune.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.trial_batch_size = trial_batch_size
        self.sample_batch_size = sample_batch_size
        # self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        # projection head append after encoder
        self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.pretrain_n_epochs = 0
        self.pretrain_n_iters = 0

        self.finetune_n_epochs = 0
        self.finetune_n_iters = 0
    
    def fit(self, trial_train_data, trial_train_labels, sample_data_length=256, overlapping=0.0, masks=None, factors=None, n_epochs=None, n_iters=None, verbose=True):
        ''' Training the TS2Vec model.
        
        Args:
            trial_train_data (numpy.ndarray): The trial-level training data. It should have a shape of (n_trials, n_timestamps, n_features). All missing data should be set to NaN.
            trial_train_labels (numpy.ndarray): The trial-level training labels. It should have a shape of (n_trials, 2). The first column is the label and the second column is patient ID.
            sample_data_length (int): The length of timestamps for sample-level data.
            overlapping (float): The overlapping for sample level data.
            masks (list): A list of mask (str). [Patient, Trial, Sample, Observation].
            factors (list): A list of loss factors. [Patient, Trial, Sample, Observation].
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert trial_train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if trial_train_data.size <= 100000 else 600  # default param for n_iters

        """if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)"""

        """temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]"""

        # we need patient ID for patient-level contrasting
        train_dataset = TensorDataset(torch.from_numpy(trial_train_data).to(torch.float), torch.from_numpy(trial_train_labels).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.trial_batch_size, len(train_dataset)), shuffle=True, drop_last=False)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        epoch_loss_list, epoch_f1_list = [], []
        
        while True:
            # count by epoch
            if n_epochs is not None and self.pretrain_n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for x, pid in train_loader:
                # count by iterations
                if n_iters is not None and self.pretrain_n_iters >= n_iters:
                    interrupted = True
                    break
                
                # x = batch[0]
                """if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]"""
                x, pid_tid = split_data_label(x.numpy(), pid.numpy(), sample_data_length, overlapping)
                x = torch.from_numpy(x).to(self.device)
                pid = torch.from_numpy(pid_tid[:, 1]).to(self.device)  # patient id
                tid = torch.from_numpy(pid_tid[:, 2]).to(self.device)  # trial id

                optimizer.zero_grad()

                # positive pairs construction
                # the fixed random seed guarantee reproducible
                # but does not mean the same mask will generate same result in one running

                if masks is None:
                    masks = ['all_true', 'all_true', 'continuous', 'continuous']

                patient_out1 = self._net(x, mask=masks[0])
                patient_out2 = self._net(x, mask=masks[0])

                trial_out1 = self._net(x, mask=masks[1])
                trial_out2 = self._net(x, mask=masks[1])

                sample_out1 = self._net(x, mask=masks[2])
                sample_out2 = self._net(x, mask=masks[2])

                observation_out1 = self._net(x, mask=masks[3])
                observation_out2 = self._net(x, mask=masks[3])
                # print(out1.shape)

                if factors is None:
                    factors = [1.0, 1.0, 1.0, 1.0]
                # loss calculation
                patient_loss = contrastive_loss(
                    patient_out1,
                    patient_out2,
                    patient_contrastive_loss,
                    id=pid,
                    hierarchical=False,
                    factor=factors[0],
                )
                trial_loss = contrastive_loss(
                    trial_out1,
                    trial_out2,
                    trial_contrastive_loss,
                    id=tid,
                    hierarchical=False,
                    factor=factors[1],
                )
                sample_loss = contrastive_loss(
                    sample_out1,
                    sample_out2,
                    sample_contrastive_loss,
                    hierarchical=True,
                    factor=factors[2],
                )
                observation_loss = contrastive_loss(
                    observation_out1,
                    observation_out2,
                    observation_contrastive_loss,
                    hierarchical=True,
                    factor=factors[3],
                )

                loss = patient_loss + trial_loss + sample_loss + observation_loss

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.pretrain_n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            epoch_loss_list.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.pretrain_n_epochs}: loss={cum_loss}")
            self.pretrain_n_epochs += 1
            
            if self.after_epoch_callback is not None:
                linear_f1 = self.after_epoch_callback(self, cum_loss)
                epoch_f1_list.append(linear_f1)
            
        return epoch_loss_list, epoch_f1_list
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        # full_series - do max pooling to the full time stamps
        # from B x T x Ch to B x 1 x Ch
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        # return out.cpu()
        return out
    
    def encode(self, data, mask=None, encoding_window=None, sample_batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): The sample-level data. This should have a shape of (n_samples, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            sample_batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if sample_batch_size is None:
            sample_batch_size = self.sample_batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=sample_batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                # print(next(self.net.parameters()).device)
                # print(x.device)
                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        # return output.numpy()
        return output.cpu().numpy()

    def finetune_fit(self, train_data, train_labels, test_data, test_labels, mask=None, encoding_window=None, sample_batch_size=None, finetune_epochs=20, finetune_lr=0.001, fraction=None):
        ''' Compute representations using the model.
        
        Args:
            train_data (numpy.ndarray): This should have a shape of (n_samples, n_timestamps, n_features). All missing data should be set to NaN.
            test_data (numpy.ndarray): This should have a shape of (n_samples, n_timestamps, n_features). All missing data should be set to NaN.
            train_labels (numpy.ndarray): This should have a shape of (n_samples,2). The first column is the label and the second column is the patient ID.
            test_labels (numpy.ndarray): This should have a shape of (n_samples,2). The first column is the label and the second column is the patient ID.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'channel-binomial', 'continuous', 'channel-continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            sample_batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            fraction (Union[float, NoneType]): The fraction of training data. It used to do semi-supervised learning.
            
        Returns:
            repr: The representations for data.
        '''

        # semi-supervised learning
        if fraction:
            train_data = train_data[:int(train_data.shape[0] * fraction)]
            train_labels = train_labels[:int(train_labels.shape[0] * fraction)]

        assert self.net is not None, 'please train or load a net first'
        assert train_data.ndim == 3
        if sample_batch_size is None:
            sample_batch_size = self.sample_batch_size
        # n_samples, ts_l, _ = train_data.shape

        self.net.train()
        self.proj_head.train()
        
        # (n_samples,) is mapped to a target of shape (n_samples, n_classes). Use for calc AUROC
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), F.one_hot(torch.from_numpy(train_labels).to(torch.long), num_classes=2).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=sample_batch_size)

        # optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        criterion = nn.CrossEntropyLoss()
        
        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_f1_list = [], [], []

        for epoch in range(finetune_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims

                y_pred = self.proj_head(out).squeeze(1)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                proj_head_optimizer.step()
                # self.net.update_parameters(self._net)
                iter_loss_list.append(loss.item())

                self.finetune_n_iters += 1

            epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))

            print(f"Epoch number: {epoch}")
            print(f"Loss: {epoch_loss_list[-1]}")
            # print(f"Accuracy: {accuracy}")
            # No mask when testing
            f1 = self.finetune_predict(test_data, test_labels, mask=None, encoding_window=encoding_window)
            epoch_f1_list.append(f1)
            self.finetune_n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, f1, fraction)
        
        return epoch_loss_list, epoch_f1_list
            
        # self.net.train(org_training)
        # return output.numpy()
        # return output.cpu().numpy()

    def finetune_predict(self, test_data, test_labels, mask=None, encoding_window=None, batch_size=None):
        test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),F.one_hot(torch.from_numpy(test_labels).to(torch.long),num_classes=2).to(torch.float))
        if batch_size is None:
            batch_size = self.sample_batch_size
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        org_training = self.net.training
        self.net.eval()
        self.proj_head.eval()

        sample_num = len(test_data)
        y_pred_prob_all = torch.zeros((sample_num, 2))
        y_target_prob_all = torch.zeros((sample_num, 2))

        with torch.no_grad():
            for index, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)  # y : (b,n_classes)

                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims

                y_pred_prob = self.proj_head(out).squeeze(1).cpu()  # (b,n_classes)
                y_target_prob = y.cpu()  # (b,n_classes)
                y_pred_prob_all[index*batch_size:index*batch_size+len(y)] = y_pred_prob
                y_target_prob_all[index*batch_size:index*batch_size+len(y)] = y_target_prob

            y_pred = y_pred_prob_all.argmax(dim=1)  # (n,)
            y_target = y_target_prob_all.argmax(dim=1)  # (n,)
            metrics_dict = {}
            metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(y_target, y_pred)
            metrics_dict['Precision'] = sklearn.metrics.precision_score(y_target, y_pred, average='macro')
            metrics_dict['Recall'] = sklearn.metrics.recall_score(y_target, y_pred, average='macro')
            metrics_dict['F1'] = sklearn.metrics.f1_score(y_target, y_pred, average='macro')
            metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y_target_prob_all, y_pred_prob_all, multi_class='ovr')
            metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y_target_prob_all, y_pred_prob_all)
            print(metrics_dict)

        self.net.train(org_training)
        self.proj_head.train(org_training)

        return metrics_dict['F1']

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)

    def finetune_save(self, encoder_fn, proj_head_fn):
        ''' Save the model to a file.

        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), encoder_fn)
        torch.save(self.proj_head.state_dict(), proj_head_fn)

    def finetune_load(self, encoder_fn, proj_head_fn):
        ''' Load the model from a file.

        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        encoder_state_dict = torch.load(encoder_fn)
        proj_head_state_dict = torch.load(proj_head_fn)
        self.net.load_state_dict(encoder_state_dict)
        self.proj_head.load_state_dict(proj_head_state_dict)
    
