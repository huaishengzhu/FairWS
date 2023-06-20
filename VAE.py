import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import random
EPS = 1e-12


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))



class VAE_fair(torch.nn.Module):
    def __init__(self, feat_d, feat_related_d, label_dim, hidden_dim, z_dim, a_dim, temperature=1, feat_related_g=None, custom=False):
        super(VAE_fair, self).__init__()
        self.encoder_A = Encoder(feat_related_d + label_dim, hidden_dim, hidden_dim)
        self.encoder_Z = Encoder(feat_related_d + feat_d + label_dim, hidden_dim, hidden_dim)
        if feat_related_g is not None:
            self.decoder_r = Decoder(z_dim + a_dim, hidden_dim, feat_related_g)
            # self.linear = nn.Sequential(
            #     nn.Linear(2*feat_related_g, feat_related_g),
            #     nn.ReLU(),
            #     nn.Linear(feat_related_g, 1),
            #     nn.Sigmoid())
        else:
            self.decoder_r = Decoder(z_dim + a_dim, hidden_dim, feat_related_d)
            # self.linear = nn.Sequential(
            #     nn.Linear(feat_related_d * 2, feat_related_d),
            #     nn.ReLU(),
            #     nn.Linear(feat_related_d, 1),
            #     nn.Sigmoid())

        self.decoder_z = Decoder(z_dim, hidden_dim, feat_d)
        self.decoder_y = Decoder(z_dim + a_dim, hidden_dim, label_dim)
        self.custom = custom
        # self._enc_A = torch.nn.Linear(hidden_dim, a_dim)
        # self._enc_log_sigma_A = torch.nn.Linear(hidden_dim, z_dim)
        # self._enc_A = torch.nn.Linear(hidden_dim, z_dim)
        self.temperature = temperature

        self._enc_mu_Z = torch.nn.Linear(hidden_dim, z_dim)
        self._enc_log_sigma_Z = torch.nn.Linear(hidden_dim, z_dim)

        self._enc_mu_A = torch.nn.Linear(hidden_dim, a_dim)
        self._enc_log_sigma_A = torch.nn.Linear(hidden_dim, a_dim)

    def _sample_latent(self, h_enc, _enc_mu, _enc_log_sigma, mode='test'):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = _enc_mu(h_enc)

        log_sigma = _enc_log_sigma(h_enc)
        if self.custom:
            sigma = torch.exp(log_sigma)
        else:
            sigma = torch.exp(F.normalize(log_sigma, dim=1))
        # if torch.isinf(torch.mean(sigma)) or torch.mean(sigma)>=10e6:
        #     sigma = torch.exp(F.normalize(log_sigma, dim=1))
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(mu)
        if mode == 'train':
            return mu + sigma * Variable(std_z, requires_grad=False)
        return mu + sigma * Variable(std_z, requires_grad=False), mu, sigma  # Reparameterization trick

    def sample_gumbel_softmax(self, state_A, mode = 'train'):
        # alpha = self._enc_A(state_A)
        alpha = F.softmax(self._enc_A(state_A), dim=1)
        unif = torch.rand(alpha.size()).to(alpha)
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + EPS)
        logit = F.softmax((log_alpha + gumbel) / self.temperature, dim=1)

        if mode == 'train':
            return logit, alpha
            # return F.gumbel_softmax(alpha), alpha

        else:
            _, max_alpha = torch.max(logit, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            one_hot_samples = one_hot_samples.to(alpha)
            return one_hot_samples


    def sampling(self, state_A, state_Z):
        Z, mu_Z, sigma_Z = self._sample_latent(state_Z, self._enc_mu_Z, self._enc_log_sigma_Z)
        A, mu_A, sigma_A = self._sample_latent(state_A, self._enc_mu_A, self._enc_log_sigma_A)
        self.mean = mu_Z
        self.sigma = sigma_Z
        self.mean_A = mu_A
        self.sigma_A =sigma_A
        return A, Z

    def inference(self, state_A):
        h_enc_A = self.encoder_A(state_A)
        A = self._sample_latent(h_enc_A, self._enc_mu_A, self._enc_log_sigma_A, mode='train')
        return A.detach()

    def forward(self, state_A, state_Z, k=None):
        h_enc_A = self.encoder_A(state_A)
        h_enc_Z = self.encoder_Z(state_Z)


        A, Z = self.sampling(h_enc_A, h_enc_Z)


        X_r = self.decoder_r(torch.cat((A, Z), dim=1))
        X_z = self.decoder_z(Z)
        y = self.decoder_y(torch.cat((A, Z), dim=1))
        # y = F.softmax(y, dim=1)
        if k is not None:
            # print(X_r[0:k,:].shape, X_r[k:2*k,:].shape)
            pos_emb = torch.cat((X_r[0:k,:],X_r[k:2*k,:]), dim=1)
            neg_emb = torch.cat((X_r[2*k:3*k, :],X_r[3*k:4*k, :]), dim=1)

            # pos_emb = (X_r[0:k, :] + X_r[k:2 * k, :])/2
            # neg_emb = (X_r[2*k:3*k, :] + X_r[3*k:4*k, :]) / 2
            adj_pred = self.linear(torch.cat((pos_emb, neg_emb), dim=0))

            # return X_r[4*k:,:], X_z[4*k:,:], y[4*k:,:], A_logits[4*k:,:], adj_pred, torch.cat((torch.ones(k), torch.zeros(k)), dim =0).to(X_r)

        return X_r, X_z, y

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
    # return 0.5 * torch.mean(mean_sq + z_stddev - torch.log(z_stddev) - 1)


def latent_loss_discrete(alpha):
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)]).to(alpha)
    log_ratio = torch.log(alpha*disc_dim+1e-20)
    loss = torch.sum(alpha * log_ratio, dim=-1).mean()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return loss



