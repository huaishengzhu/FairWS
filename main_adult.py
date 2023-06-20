import argparse
import random
import torch
import numpy as np
from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import PandasDataSet
from torch.utils.data import DataLoader
from VAE import VAE_fair, latent_loss, latent_loss_discrete
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from fair_model import Classifier, Classifier_lr, loss_SVM, pretrain_classifier, CorreErase_train, Predictor, Adversarial, Adversarial_train
from fairlearn.metrics import MetricFrame
from sklearn import metrics
from utils import sens_to_int
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture

def train(model, optimizer, epochs, dataloader, criterion):
    pre_loss = np.inf
    count = 0
    for epoch in range(epochs):
        total_loss = []
        recons_y = []
        recons_f = []
        for i, data in enumerate(dataloader, 0):
            inputs, inputs_related, y, indexs = data
            # print(inputs.shape, inputs_related.shape)
            optimizer.zero_grad()
            inputs_A = torch.cat((inputs_related, y), dim = 1)
            # inputs_Z = torch.cat((inputs, y), dim=1)
            inputs_Z = torch.cat((torch.cat((inputs_related, inputs), dim = 1), y), dim=1)
            X_r, X_z, X_y = model(inputs_A, inputs_Z)
            ll_Z = latent_loss(model.mean, model.sigma)
            ll_A = latent_loss(model.mean_A, model.sigma_A)
            inputs_loss = torch.cat((inputs_related, inputs), dim =1)
            y = torch.nonzero(y)[:, 1]
            criterion1 = nn.CrossEntropyLoss()
            ll_y = criterion1(X_y, y.long())
            ll_f = criterion(X_r, inputs_related) + criterion(X_z, inputs)
            # ll_f = criterion(torch.cat((X_r, X_z), dim = 1), inputs_loss)
            recons_y.append(ll_A.item())
            recons_f.append(ll_f.item())
            loss = ll_f + ll_y + 0.01 * ll_A + ll_Z
            loss.backward()
            optimizer.step()
            l = loss.item()
            total_loss.append(l)

        acc = test_acc(model, train_loader, y_true, sens_vectors)
        print("Epoch:{:04d} ".format(epoch+1)
              + "Loss:{:.2f} ".format(sum(total_loss)/len(total_loss))
              +"Loss_A:{:.2f} ".format(sum(recons_y)/len(recons_y))
              +"Loss_f:{:.2f} ".format(sum(recons_f)/len(recons_f)))
        # if pre_loss > sum(recons_f)/len(recons_f):
        #     pre_loss = sum(recons_f)/len(recons_f)
        # else:
        #     break


def test_acc(model, dataloader, y_train, sens_attr, last = False):
    pred_list = []
    index_list = []
    for i, data in enumerate(dataloader, 0):
        model.eval()
        inputs, inputs_related, y, indexs = data
        inputs_A = torch.cat((inputs_related, y), dim=1)
        discrete_A = model.inference(inputs_A)
        pred_list += discrete_A.cpu().numpy().tolist()
        index_list += indexs.cpu().numpy().astype(int).tolist()


    _, true_sens = np.where(np.array(sens_attr)[index_list]==1)
    GM = GaussianMixture(n_components=2).fit(pred_list)
    pred = GM.predict_proba(pred_list)

    acc_pos = roc_auc_score(true_sens, pred[:, 1])
    acc_neg = roc_auc_score(true_sens, 1 - pred[:, 1])
    # print(acc)
    if last == True:
        return max(acc_neg, acc_pos), GM
    else:
        return max(acc_neg, acc_pos)

if __name__ == '__main__':
    # adult --related age relationship marital-status
    # law --related Race Year resident
    # compas --related sex age duration
    # 0.04 for adult
    parser = argparse.ArgumentParser(description='FairVAE')
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--method", default="remove", type=str,
                        choices=['base', 'corre', 'groupTPR', 'learn', 'remove', 'learnCorre'])
    parser.add_argument("--dataset", default="adult", type=str, choices=['adult', 'pokec', 'compas', 'law', 'movielens'])
    parser.add_argument("--s", default="sex", type=str)  # sex for adult
    parser.add_argument("--related", nargs='+',
                        type=str)  # choices=['sex','race','age','relationship','marital-status', 'education', 'workclass'] for adult
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--beta", default=0.5, type=float)  # weight for regularization of Lambda

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--hidden_dim", default=8, type=int)

    parser.add_argument("--hidden_dim_classifier", default=64, type=int)

    parser.add_argument("--latent_hidden_dim", default=8, type=int)

    parser.add_argument("--sens_dim", default=12, type=int)

    parser.add_argument("--batch_size", default=320, type=int)

    parser.add_argument("--model", default='MLP', type=str,
                        choices=['MLP', 'LR', 'SVM', 'ADV'])  # weight for regularization of Lambda
    parser.add_argument("--pretrain_epoch", default=1, type=int)

    parser.add_argument("--epoch_classifier", default=10, type=int)

    parser.add_argument("--corre_weight", default=0.017, type=float)

    parser.add_argument("--weight", default=0.3, type=float)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    sens = False
    if sens:
        args.epoch = 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    X, X_related, y_true, sensitive_attr, n_classes = load_data(args)
    sens_vectors = sens_to_int(sensitive_attr)
    # print(sensitive_attr)
    indict = np.arange(sensitive_attr.shape[0])
    (X_train, X_test, X_related_train, X_related_test, y_train, y_test, ind_train, ind_test) = train_test_split(X, X_related, y_true, indict, test_size=0.5,
                                                                               stratify=y_true, random_state=7)
    processed_X_train = X_train
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = StandardScaler().fit(X_related_train)
    X_related_train = scaler.transform(X_related_train)
    X_related_test = scaler.transform(X_related_test)

    # args.sens_dim = sens_vectors.shape[1]


    train_data = PandasDataSet(X_train, X_related_train, y_train, ind_train, device=device)
    test_data = PandasDataSet(X_test, X_related_test, y_test, ind_test, device=device)
    train_VAE_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print('# training samples:', len(train_data))
    print('# batches:', len(train_loader))

    n_features = X.shape[1]
    n_features_related = X_related.shape[1]
    label_dim = n_classes
    # label_dim = y_true.shape[1]

    input_dim = n_features + n_features_related

    model = VAE_fair(n_features, n_features_related, label_dim, args.hidden_dim, args.latent_hidden_dim, args.sens_dim, custom=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train(model, optimizer, args.epoch, train_VAE_loader, criterion)

    acc, GM = test_acc(model, train_loader, y_true, sens_vectors, last = True)

    if args.model == 'MLP':
        clf = Classifier(n_features=input_dim, n_hidden=args.hidden_dim_classifier, n_class=n_classes)
    elif args.model == 'LR':
        clf = Classifier_lr(n_features=input_dim, n_class=n_classes)
    elif args.model == 'SVM':
        assert n_classes == 2, "classes need to be 2 for SVM classifier"
        clf = Classifier_lr(n_features=input_dim, n_class=1)
    elif args.model == 'ADV':
        predictor = Predictor(n_features=input_dim, n_class=n_classes)
        adversary = Adversarial(n_features=n_classes, n_class=sens_vectors.shape[1])
    else:
        raise NotImplementedError("not implemented model: {}".format(args.model))
    if args.model != 'ADV':
        clf.to(device)
        clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)
    else:
        predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
        adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=args.lr)
        predictor.to(device)
        adversary.to(device)

    if args.model != 'SVM':
        clf_criterion = nn.CrossEntropyLoss()
    else:
        clf_criterion = loss_SVM
    if args.model != 'ADV':
        for i in range(args.pretrain_epoch):
            clf = clf.train()
            clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion, args.model)
        for epoch in range(args.epoch_classifier):
            clf = clf.train()
            if sens:
                clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, model, args.model, args.corre_weight, GM, sens=torch.IntTensor(sens_vectors).to(device))
            else:
                clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, model, args.model, args.corre_weight, GM)
    else:
        for i in range(args.pretrain_epoch):
            predictor = predictor.train()
            predictor = pretrain_classifier(predictor, train_loader, predictor_optimizer, clf_criterion, args.model)
        for epoch in range(args.epoch_classifier):
            predictor.train()
            adversary.train()
            if sens:
                clf = Adversarial_train(predictor, adversary, train_loader, predictor_optimizer, adversary_optimizer, clf_criterion, model, args.weight, sens=torch.IntTensor(sens_vectors).to(device))
            else:
                clf = Adversarial_train(predictor, adversary, train_loader, predictor_optimizer, adversary_optimizer, clf_criterion, model, args.weight)
        # pre_clf_test = clf(torch.cat((test_data.tensors[0], test_data.tensors[1]), dim=1))
    # Test
    with torch.no_grad():
        pre_clf_test = clf(torch.cat((test_data.tensors[0], test_data.tensors[1]), dim =1))
    if args.model != 'SVM':
        y_pred = pre_clf_test.argmax(dim=1).cpu()
    else:
        y_pred = (pre_clf_test > 0).reshape(-1).int().cpu()
    _, true_sens_test = np.where(np.array(sens_vectors)[ind_test] == 1)
    y_test = np.argwhere(y_test)[:,1]
    gm = MetricFrame(metrics.accuracy_score, y_test, y_pred, sensitive_features=true_sens_test)
    print('Average accuracy score: {}'.format(gm.overall))
    print(gm.by_group)

    group_selection_rate = []
    group_equal_odds = []
    sens_test = true_sens_test
    for sens_value in set(sens_test):
        y_sense_pred = y_pred[(sens_test == sens_value)]
        y_sense_test = y_test[(sens_test == sens_value)]
        sens_sr = []
        sens_eo = []

        for label in set(y_test):
            if label > 0:
                sens_sr_label = (y_sense_pred == label).sum() / y_sense_pred.shape[0]
                sens_eo_label = (y_sense_pred[y_sense_test == label] == label).sum() / (y_sense_test == label).sum()

                sens_sr.append(sens_sr_label)
                sens_eo.append(sens_eo_label)

        group_selection_rate.append(sens_sr)
        group_equal_odds.append(sens_eo)

    group_selection_rate = np.array(group_selection_rate)
    group_equal_odds = np.array(group_equal_odds)

    print('group equal odds: ')
    print(group_equal_odds)
    print('eo_difference: {}'.format(
        np.mean(np.absolute(group_equal_odds - np.mean(group_equal_odds, axis=0, keepdims=True)))))
    if args.dataset == 'compas':
        print('target eo_difference: {}'.format((np.absolute(group_equal_odds[0] - group_equal_odds[2]))))

    print('group selection rate: ')
    print(group_selection_rate)
    print('sr_difference: {}'.format(
        np.mean(np.absolute(group_selection_rate - np.mean(group_selection_rate, axis=0, keepdims=True)))))












