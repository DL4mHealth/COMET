import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import sklearn


def finetune_fit(model, X_train, y_train, X_test, y_test, batch_size=128, finetune_epochs=50, num_classes=2,
                 finetune_lr=0.0001, fraction=None, device='cuda', callback=None):
    """ finetune the whole model including encoder and classifier.

    Args:
        model (nn.Module): This should be a model composed by encoder and classifier.
        X_train (numpy.ndarray): This should have a shape of (n_samples, sample_timestamps, features)
        X_test (numpy.ndarray): This should have a shape of (n_samples, sample_timestamps, features)
        y_train (numpy.ndarray): This should have a shape of (n_samples,).
        y_test (numpy.ndarray): This should have a shape of (n_samples,).
        finetune_epochs (int): The number of epochs for fine-tuning.
        num_classes (int): The number of label classes.
        finetune_lr (float): The learning rate for fine-tuning.
        batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        fraction (Union[float, NoneType]): The fraction of training data. It used to do semi-supervised learning.
        device (str): The gpu used for training and inference.
        callback (Union[func, NoneType]): A callback function that would be called after each epoch.

    Returns:
        repr: list of epoch loss and f1
    """

    # semi-supervised learning: only use fraction of training samples.
    if fraction:
        X_train = X_train[:int(X_train.shape[0] * fraction)]
        y_train = y_train[:int(y_train.shape[0] * fraction)]

    assert X_train.ndim == 3
    device = torch.device(device)

    model.train()

    # (n_samples,) is mapped to a target of shape (n_samples, n_classes).
    train_dataset = TensorDataset(torch.from_numpy(X_train).to(torch.float),
                                  F.one_hot(torch.from_numpy(y_train).to(torch.long),
                                            num_classes=num_classes).to(torch.float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

    # implicitly implement softmax function in cross entropy loss
    criterion = nn.CrossEntropyLoss()

    epoch_loss_list, iter_loss_list, epoch_f1_list = [], [], []
    model.n_epochs = 1

    for epoch in range(finetune_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            iter_loss_list.append(loss.item())

        epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {epoch_loss_list[-1]}")
        # print(f"Accuracy: {accuracy}")
        # No mask when testing
        f1 = finetune_predict(model, X_test, y_test, batch_size=batch_size, num_classes=num_classes, device=str(device))
        epoch_f1_list.append(f1)
        callback(model, f1, fraction)

        model.n_epochs += 1

    return epoch_loss_list, epoch_f1_list


def finetune_predict(model, X_test, y_test, batch_size=128, num_classes=2, device='cuda'):
    """ test the fine-tuned model

    Args:
        model (nn.Module): This should be a model composed by encoder and classifier.
        X_test (numpy.ndarray): This should have a shape of (n_samples, sample_timestamps, features)
        y_test (numpy.ndarray): This should have a shape of (n_samples,).
        batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        num_classes (int): The number of label classes.
        device (str): The gpu used for training and inference.

    Returns:
        repr: f1 score
    """
    device = torch.device(device)

    test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float),
                                 F.one_hot(torch.from_numpy(y_test).to(torch.long), num_classes=num_classes).to(torch.float))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    org_training = model.training
    model.eval()

    sample_num = len(X_test)
    y_pred_prob_all = torch.zeros((sample_num, num_classes))
    y_target_prob_all = torch.zeros((sample_num, num_classes))

    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)  # y : (B,n_classes)

            y_pred_prob = model(x).cpu()  # (B, n_classes)
            y_target_prob = y.cpu()  # (B, n_classes)
            y_pred_prob_all[index*batch_size:index*batch_size+len(y)] = y_pred_prob
            y_target_prob_all[index*batch_size:index*batch_size+len(y)] = y_target_prob

        y_pred = y_pred_prob_all.argmax(dim=1)  # (B, )
        y_target = y_target_prob_all.argmax(dim=1)  # (b, )
        metrics_dict = {}
        metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(y_target, y_pred)
        metrics_dict['Precision'] = sklearn.metrics.precision_score(y_target, y_pred, average='macro')
        metrics_dict['Recall'] = sklearn.metrics.recall_score(y_target, y_pred, average='macro')
        metrics_dict['F1'] = sklearn.metrics.f1_score(y_target, y_pred, average='macro')
        metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y_target_prob_all, y_pred_prob_all, multi_class='ovr')
        metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y_target_prob_all, y_pred_prob_all)
        print(metrics_dict)

    model.train(org_training)

    return metrics_dict['F1']
