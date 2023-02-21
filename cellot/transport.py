import anndata
from torch.utils.data import DataLoader


def transport(config, model, dataset, return_as="anndata", dosage=None, **kwargs):
    name = config.model.get("name", "cellot")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    if name == "cellot":
        outputs = transport_cellot(model, inputs)

    elif name == "scgen":
        outputs = transport_scgen(
            model,
            inputs,
            source=config.data.source,
            target=config.data.target,
            **kwargs
        )

    elif name == "cae":
        outputs = transport_cae(model, inputs, target=config.data.target)

    else:
        raise ValueError

    if dosage is not None:
        outputs = (1 - dosage) * inputs + dosage * outputs

    if return_as == "anndata":
        try:
            item = outputs.detach().numpy()
        except AttributeError:
            item = outputs

        outputs = anndata.AnnData(
            item,
            obs=dataset.adata.obs.copy(),
            var=dataset.adata.var.copy(),
        )

    return outputs


def transport_cellot(model, inputs):
    f, g = model
    g.eval()
    outputs = g.transport(inputs.requires_grad_(True))
    return outputs


def transport_scgen(model, inputs, source, target, decode=True):
    model.eval()
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(inputs)
    if not decode:
        return codes + shift

    outputs = model.decode(codes + shift)
    return outputs


def transport_cae(model, inputs, target):
    model.eval()
    target_code = model.conditions.index(target)
    outputs = model.outputs(inputs, decode_as=target_code).recon
    return outputs
