import torch
from torch import nn
from collections import namedtuple
from pathlib import Path
from torch.utils.data import DataLoader


def load_optimizer(config, params):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    optim = torch.optim.Adam(params, **kwargs)
    return optim


def load_networks(config, **kwargs):
    kwargs = kwargs.copy()
    kwargs.update(dict(config.get("model", {})))
    name = kwargs.pop("name")

    if name == "scgen":
        model = AutoEncoder

    elif name == "cae":
        model = ConditionalAutoEncoder

    else:
        raise ValueError

    return model(**kwargs)


def load_autoencoder_model(config, restore=None, **kwargs):
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if config.model.name == "scgen" and "code_means" in ckpt:
            model.code_means = ckpt["code_means"]

    return model, optim


def dnn(
    dinput,
    doutput,
    hidden_units=(16, 16),
    activation="ReLU",
    dropout=0.0,
    batch_norm=False,
    net_fn=nn.Sequential,
    **kwargs
):

    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)

    layer_sizes = zip([dinput] + hidden_units[:-1], hidden_units)

    if isinstance(activation, str):
        Activation = getattr(nn, activation)
    else:
        Activation = activation

    layers = list()
    for indim, outdim in layer_sizes:
        layers.append(nn.Linear(indim, outdim, **kwargs))

        if batch_norm:
            layers.append(nn.BatchNorm1d(outdim))

        layers.append(Activation())

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_units[-1], doutput))
    net = nn.Sequential(*layers)
    return net


class AutoEncoder(nn.Module):
    LossComps = namedtuple("AELoss", "mse reg")
    Outputs = namedtuple("AEOutputs", "recon code")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=None,
        beta=0,
        dropout=0,
        mse=None,
        **kwargs
    ):

        super(AutoEncoder, self).__init__(**kwargs)

        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        if decoder_net is None:
            assert hidden_units is not None
            decoder_net = self.build_decoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        if mse is None:
            mse = nn.MSELoss(reduction="none")

        self.mse = mse

        return

    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            dinput=input_dim, doutput=latent_dim, hidden_units=hidden_units, **kwargs
        )
        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            dinput=latent_dim,
            doutput=input_dim,
            hidden_units=hidden_units[::-1],
            **kwargs
        )
        return net

    def encode(self, inputs, **kwargs):
        return self.encoder_net(inputs, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def outputs(self, inputs, **kwargs):
        code = self.encode(inputs, **kwargs)
        recon = self.decode(code, **kwargs)
        outputs = self.Outputs(recon, code)
        return outputs

    def loss(self, inputs, outputs):
        mse = self.mse(outputs.recon, inputs).mean(dim=-1)
        reg = torch.norm(outputs.code, dim=-1) ** 2
        total = mse + self.beta * reg
        comps = self.LossComps(mse, reg)
        return total, comps

    def forward(self, inputs, **kwargs):
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(inputs, outs)

        return loss, comps, outs


def compute_scgen_shift(model, dataset, labels):
    model.code_means = dict()

    inputs = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    codes = model.encode(inputs)

    for key in labels.unique():
        mask = labels == key
        model.code_means[key] = codes[mask.values].mean(0)

    return model.code_means


class ConditionalAutoEncoder(AutoEncoder):
    def __init__(self, *args, conditions, **kwargs):
        self.conditions = conditions
        self.n_cats = len(conditions)
        super(ConditionalAutoEncoder, self).__init__(*args, **kwargs)
        return

    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = super().build_encoder(
            input_dim=input_dim + self.n_cats,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            **kwargs
        )

        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = super().build_decoder(
            input_dim=input_dim,
            latent_dim=latent_dim + self.n_cats,
            hidden_units=hidden_units,
            **kwargs
        )

        return net

    def condition(self, data, labels):
        conds = nn.functional.one_hot(labels, self.n_cats)
        return torch.cat([data, conds], dim=1)

    def encode(self, inputs, **kwargs):
        data, labels = inputs
        cond = self.condition(data, labels)
        return self.encoder_net(cond)

    def decode(self, codes, **kwargs):
        data, labels = codes
        cond = self.condition(data, labels)
        return self.decoder_net(cond)

    def outputs(self, inputs, decode_as=None, **kwargs):
        data, label = inputs
        assert len(data) == len(label)

        decode_label = label if decode_as is None else decode_as
        if isinstance(decode_label, str):
            raise NotImplementedError

        if isinstance(decode_label, int):
            decode_label = decode_label * torch.ones(len(data), dtype=int)

        code = self.encode((data, label), **kwargs)
        recon = self.decode((code, decode_label), **kwargs)
        outputs = self.Outputs(recon, code)
        return outputs

    def forward(self, inputs, beta=None, **kwargs):
        values, _ = inputs
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(values, outs)

        return loss, comps, outs
