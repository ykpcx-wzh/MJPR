# --------------------------------------------------------
# This code is borrowed from https://github.com/facebookresearch/deep_bisim4control/blob/main/transition_model.py
# --------------------------------------------------------

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Attention(nn.Module):
    def __init__(self,
                 dim=50,  # 输入token的dim 768
                 num_heads=5,  # multi-head 12
                 qkv_bias=False,  # True
                 qk_scale=None,  # 和根号dimk作用相同

                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768 // 12 = 64
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv经过一个linear得到。 768 --> 2304
        self.proj = nn.Linear(dim, dim)  # 一次新的映射
        self.act=nn.ReLU()

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.act(qkv)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 在Vit中qkv维度相同，都是[B,12,197,64]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 按行进行softmax
        attn = attn.softmax(dim=-1)


        #print((attn @ v).shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 再通过一次映射使其更好的融合
        
        x = self.proj(x)
        x = self.act(x)
	
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features  # 768
        hidden_features = hidden_features or in_features  # 3072
        self.fc1 = nn.Linear(in_features, hidden_features)  # 768 --> 3072
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 3072 --> 768

    def forward(self, x,mask=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(50)

    def forward(self, x, **kwargs):
        res = x
        x=self.ln(x)
        x = self.fn(x, **kwargs)
        x += res
        return x


class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.at1 = ResidualAdd(Attention())
        self.at2 = ResidualAdd(Attention())
       
        #self.at3 = Attention()
        #self.mlp1 = ResidualAdd(Mlp(in_features=50,hidden_features=128))

    def forward(self,x):

        x=self.at1(x)
        x=self.at2(x)
        #x=self.at3(x)
        
        return x

class DeterministicTransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape, layer_width):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim*2 + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        print("Deterministic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu

class TransitionModel(nn.Module):
    def __init__(self, encoder_feature_dim, action_shape):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], encoder_feature_dim)

        print("Deterministic transition model chosen.")

    def forward(self, x):


        mu = self.fc(x)

        return mu


class ProbabilisticTransitionModel(nn.Module):
    def __init__(self,
                 encoder_feature_dim,
                 action_shape,
                 layer_width,
                 announce=True,
                 max_sigma=1e1,
                 min_sigma=1e-4):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert (self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (
            self.max_sigma -
            self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class EnsembleOfProbabilisticTransitionModels(object):
    def __init__(self,
                 encoder_feature_dim,
                 action_shape,
                 layer_width,
                 ensemble_size=5):
        self.models = [
            ProbabilisticTransitionModel(encoder_feature_dim,
                                         action_shape,
                                         layer_width,
                                         announce=False)
            for _ in range(ensemble_size)
        ]
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self):
        list_of_parameters = [
            list(model.parameters()) for model in self.models
        ]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


_AVAILABLE_TRANSITION_MODELS = {
    '': DeterministicTransitionModel,
    'deterministic': DeterministicTransitionModel,
    'probabilistic': ProbabilisticTransitionModel,
    'ensemble': EnsembleOfProbabilisticTransitionModels
}


def make_transition_model(transition_model_type,
                          encoder_feature_dim,
                          action_shape,
                          layer_width=512):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim, action_shape, layer_width)
