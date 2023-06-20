import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_SVM(result, truth, model):
    truth[truth==0] = -1
    result = result.squeeze()
    weight = model.linear.weight.squeeze()

    loss = torch.mean(torch.clamp(1 - truth * result, min=0))
    loss += 0.1*torch.mean(torch.mul(weight, weight))

    return loss


class Classifier(nn.Module):

    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2, graph=False):
        super(Classifier, self).__init__()
        if not graph:
            self.network = nn.Sequential(
                nn.Linear(n_features, n_hidden * 2),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden * 2, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_class),
            )
        else:
            self.network = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.network(x)

class Adversarial(nn.Module):
    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
        super(Adversarial, self).__init__()
        self.adversarial = nn.Sequential(
            nn.Linear(n_features, n_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_class),
        )

        # self.adversarial = nn.Sequential(
        #     nn.Linear(n_features, n_class),
        # )


    def forward(self, x):
        sens = self.adversarial(x)
        return sens


class Predictor(nn.Module):
    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
        super(Predictor, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(n_features, n_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, x):
        pred = self.discriminator(x)
        return pred

class Classifier_lr(nn.Module):
    def __init__(self, n_features, n_class=2):
        super(Classifier_lr, self).__init__()

        self.linear = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.linear(x)

def pretrain_classifier(clf, data_loader, optimizer, criterion, model, model_GCN=None, adj=None, features= None, labels=None, text=False, emb = None):
    if model_GCN is not None:
        for i, data in enumerate(data_loader, 0):
            ind = data[0]
            inputs = features[ind]
            y = labels[ind]
            model_GCN = model_GCN.train()
            # print("Hello")
            if text is False:
                inputs_related = model_GCN(adj)[ind]
                if emb is not None:
                    inputs_related = emb[ind]
            else:
                inputs_related = model_GCN(adj[ind])
            x = torch.cat((inputs, inputs_related), dim=-1)

            # print(x.shape)
            y = torch.nonzero(y)[:, 1]
            optimizer.zero_grad()
            p_y = clf(x)
            if model != 'SVM':
                loss = criterion(p_y, y.long())
            else:
                loss = criterion(p_y, y, clf)
            # print(loss)
            loss.backward()
            optimizer.step()
    else:
        for inputs, inputs_related, y, ind in data_loader:
            x = torch.cat((inputs, inputs_related), dim=-1)
            # print(x.shape)
            y = torch.nonzero(y)[:,1]
            clf.zero_grad()
            p_y = clf(x)
            # print(p_y.shape, y.shape)
            if model != 'SVM':
                loss = criterion(p_y, y.long())
            else:
                loss = criterion(p_y, y, clf)
            loss.backward()
            optimizer.step()
    # print("sssssss")
    # return model_GCN
    if model_GCN is not None:
        return clf, model_GCN
    return clf

def CorreErase_train(clf, data_loader, optimizer, criterion, model_VAE, model, weight, GM, model_GCN=None, adj= None, sens = None, features= None, labels=None, text=False, features_norm= None, emb= None, model_GCN_clf = None):
    if model_GCN is not None:
        for ind in data_loader:
            inputs =features[ind]
            y = labels[ind]
            model_VAE.eval()
            model_GCN = model_GCN.eval()
            model_GCN_clf = model_GCN_clf.train()
            # if sens is None:
            #     model_GCN.eval()
            if text is False:
                inputs_related = model_GCN(adj)[ind]
                inputs_related_clf = model_GCN_clf(adj)[ind]
                if emb is not None:
                    inputs_related = emb[ind]
                    inputs_related_clf = emb[ind]
            else:
                inputs_related = model_GCN(adj[ind])
                inputs_related_clf = model_GCN_clf(adj[ind])
            inputs_A = torch.cat((inputs_related, y), dim=1)

            x = torch.cat((inputs, inputs_related_clf), dim=-1)
            y = torch.nonzero(y)[:, 1]
            optimizer.zero_grad()
            # model_GCN.zero_grad()
            p_y = clf(x)

            # p_y = model_GCN(adj)[ind]
            if model != 'SVM':
                loss = criterion(p_y, y.long())
            else:
                loss = criterion(p_y, y, clf)
            if sens is None:
                discrete_A = model_VAE.inference(inputs_A)
                # discrete_A = torch.FloatTensor(GM.predict(model_VAE.inference(inputs_A).cpu().numpy())).to(x)
            else:
                discrete_A = sens[ind].float()
            # discrete_A = model_VAE.inference(inputs_A)
            # discrete_A = inputs_related.detach()
            # temp = torch.eye(discrete_A.shape[1])
            # discrete_A = temp[torch.randint(0, 2, (discrete_A.shape[0],))].float().to(x)
            # for i in range(discrete_A.shape[1]):
            #     target_x = discrete_A[:, i].unsqueeze(-1)
            cor_loss = torch.sum(torch.abs(torch.mean(
                torch.mul(discrete_A.reshape(1, x.shape[0], -1) - discrete_A.mean(dim=0).reshape(1, 1, -1),
                          (p_y - p_y.mean(dim=0)).transpose(0, 1).reshape((-1, p_y.shape[0], 1))), dim=1)))

            # print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + cor_loss * weight

            loss.backward()
            optimizer.step()
    else:
        for inputs, inputs_related, y, ind in data_loader:
            ind = ind.long()
            model_VAE.eval()
            inputs_A = torch.cat((inputs_related, y), dim=1)
            x = torch.cat((inputs, inputs_related), dim=-1)
            y = torch.nonzero(y)[:,1]
            clf.zero_grad()
            p_y = clf(x)
            if model != 'SVM':
                loss = criterion(p_y, y.long())
            else:
                loss = criterion(p_y, y, clf)
            if sens is None:
                discrete_A = model_VAE.inference(inputs_A)
                # discrete_A = torch.FloatTensor(GM.predict(model_VAE.inference(inputs_A).cpu().numpy())).to(x)

            else:
                discrete_A = sens[ind].float()
            # discrete_A = inputs_related
            # temp = torch.eye(discrete_A.shape[1])
            # discrete_A = temp[torch.randint(0,2, (discrete_A.shape[0],))].float().to(x)

            # for i in range(discrete_A.shape[1]):
            #     target_x = discrete_A[:, i].unsqueeze(-1)
            cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(discrete_A.reshape(1,x.shape[0],-1) - discrete_A.mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

                #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + weight * cor_loss

            loss.backward()
            optimizer.step()
    if model_GCN is not None:
        return clf, model_GCN_clf
    return clf

def Adversarial_train(predictor, adversary, data_loader, predictor_optimizer, adversary_optimizer, criterion, model_VAE, weight, sens = None, model_GCN=None, adj= None, features= None, labels=None):
    if model_GCN is not None:
        for ind in data_loader:
            predictor_optimizer.zero_grad()
            adversary_optimizer.zero_grad()
            inputs = features[ind]
            y = labels[ind]
            model_VAE.eval()
            if sens is None:
                model_GCN.eval()
            inputs_related = model_GCN(adj)[ind]
            inputs_A = torch.cat((inputs_related, y), dim=1)
            x = torch.cat((inputs, inputs_related), dim=-1)
            y = torch.nonzero(y)[:, 1]
            pred = predictor(x)
            protect_pred = adversary(pred)
            if sens is None:
                discrete_A = model_VAE.inference(inputs_A)
            else:
                discrete_A = sens[ind]

            # discrete_A = inputs_related.detach()
            # criterion1 = nn.MSELoss()

            pred_loss = criterion(pred, y.long())
            protect_loss = criterion(protect_pred, discrete_A[:, 1].long())

            # protect_loss = criterion1(protect_pred, discrete_A)

            protect_loss.backward(retain_graph=True)
            protect_grad = {name: param.grad.clone() for name, param in predictor.named_parameters()}

            adversary_optimizer.step()

            predictor_optimizer.zero_grad()
            pred_loss.backward()

            with torch.no_grad():
                for name, param in predictor.named_parameters():
                    unit_protect = protect_grad[name] / torch.linalg.norm(protect_grad[name])
                    param.grad -= weight * ((param.grad * unit_protect) * unit_protect).sum()
                    param.grad -= weight * protect_grad[name]
            predictor_optimizer.step()
    else:
        for inputs, inputs_related, y, ind in data_loader:
            predictor_optimizer.zero_grad()
            adversary_optimizer.zero_grad()

            ind = ind.long()
            model_VAE.eval()
            inputs_A = torch.cat((inputs_related, y), dim=1)
            x = torch.cat((inputs, inputs_related), dim=-1)
            y = torch.nonzero(y)[:, 1]

            pred = predictor(x)
            protect_pred = adversary(pred)
            if sens is None:
                discrete_A = model_VAE.inference(inputs_A)
            else:
                discrete_A = sens[ind]

            pred_loss = criterion(pred, y.long())
            protect_loss = criterion(protect_pred, discrete_A[:, 1].long())

            # protect_loss = criterion1(protect_pred, discrete_A)

            protect_loss.backward(retain_graph=True)
            protect_grad = {name: param.grad.clone() for name, param in predictor.named_parameters()}

            adversary_optimizer.step()

            predictor_optimizer.zero_grad()
            pred_loss.backward()

            with torch.no_grad():
                for name, param in predictor.named_parameters():
                    unit_protect = protect_grad[name] / torch.linalg.norm(protect_grad[name])
                    param.grad -= weight * ((param.grad * unit_protect) * unit_protect).sum()
                    param.grad -= weight * protect_grad[name]
            predictor_optimizer.step()

    return predictor
