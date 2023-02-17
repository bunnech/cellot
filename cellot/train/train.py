from pathlib import Path

import torch
import numpy as np
import random
import pickle
from absl import logging
from absl.flags import FLAGS
from cellot import losses
from cellot.utils.loaders import load
from cellot.models.cellot import compute_loss_f, compute_loss_g, compute_w2_distance
from cellot.train.summary import Logger
from cellot.data.utils import cast_loader_to_iterator
from cellot.models.ae import compute_scgen_shift
from tqdm import trange


def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]


def train_cellot(outdir, config):
    def state_dict(f, g, opts, **kwargs):
        state = {
            "g_state": g.state_dict(),
            "f_state": f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
        }
        state.update(kwargs)

        return state

    def evaluate():
        target = next(iterator_test_target)
        source = next(iterator_test_source)
        source.requires_grad_(True)
        transport = g.transport(source)

        transport = transport.detach()
        with torch.no_grad():
            gl = compute_loss_g(f, g, source, transport).mean()
            fl = compute_loss_f(f, g, source, target, transport).mean()
            dist = compute_w2_distance(f, g, source, target, transport)
            mmd = losses.compute_scalar_mmd(
                target.detach().numpy(), transport.detach().numpy()
            )

        # log to logger object
        logger.log(
            "eval",
            gloss=gl.item(),
            floss=fl.item(),
            jloss=dist.item(),
            mmd=mmd,
            step=step,
        )
        check_loss(gl, gl, dist)

        return mmd

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    (f, g), opts, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)

    minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
    mmd = minmmd

    if 'pair_batch_on' in config.training:
        keys = list(iterator.train.target.keys())
        test_keys = list(iterator.test.target.keys())
    else:
        keys = None

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        if 'pair_batch_on' in config.training:
            assert keys is not None
            key = random.choice(keys)
            iterator_train_target = iterator.train.target[key]
            iterator_train_source = iterator.train.source[key]
            try:
                iterator_test_target = iterator.test.target[key]
                iterator_test_source = iterator.test.source[key]
            # in the iid mode of the ood setting,
            # train and test keys are not necessarily the same ...
            except KeyError:
                test_key = random.choice(test_keys)
                iterator_test_target = iterator.test.target[test_key]
                iterator_test_source = iterator.test.source[test_key]

        else:
            iterator_train_target = iterator.train.target
            iterator_train_source = iterator.train.source
            iterator_test_target = iterator.test.target
            iterator_test_source = iterator.test.source

        target = next(iterator_train_target)
        for _ in range(config.training.n_inner_iters):
            source = next(iterator_train_source).requires_grad_(True)

            opts.g.zero_grad()
            gl = compute_loss_g(f, g, source).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()

            gl.backward()
            opts.g.step()

        source = next(iterator_train_source).requires_grad_(True)

        opts.f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opts.f.step()
        check_loss(gl, fl)
        f.clamp_w()

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

        if step % config.training.eval_freq == 0:
            mmd = evaluate()
            if mmd < minmmd:
                minmmd = mmd
                torch.save(
                    state_dict(f, g, opts, step=step, minmmd=minmmd),
                    cachedir / "model.pt",
                )

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

            logger.flush()

    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

    logger.flush()

    return


def train_auto_encoder(outdir, config):
    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)

        return state

    def evaluate(vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)
        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")

    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf
    )

    eval_loss = best_eval_loss

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:

        model.train()
        inputs = next(iterator.train)
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training.eval_freq == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)

                torch.save(sd, cachedir / "model.pt")

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")

            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model.name == "scgen" and config.get("compute_scgen_shift", True):
        labels = loader.train.dataset.adata.obs[config.data.condition]
        compute_scgen_shift(model, loader.train.dataset, labels=labels)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")

    logger.flush()


def train_popalign(outdir, config):
    def evaluate(config, data, model):

        # Get control and treated subset of the data and projections.
        idx_control_test = np.where(data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_test = np.where(data.obs[
            config.data.condition] == config.data.target)[0]

        predicted = transport_popalign(model, data[idx_control_test].X)
        target = np.array(data[idx_treated_test].X)

        # Compute performance metrics.
        mmd = losses.compute_scalar_mmd(target, predicted)
        wst = losses.wasserstein_loss(target, predicted)

        # Log to logger object.
        logger.log(
            "eval",
            mmd=mmd,
            wst=wst,
            step=1
        )

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"

    # Load dataset and previous model parameters.
    model, _, dataset = load(config, restore=cachedir / "last.pt",
                             return_as="dataset")
    train_data = dataset["train"].adata
    test_data = dataset["test"].adata

    if not all(k in model for k in ("dim_red", "gmm_control", "response")):

        if config.model.embedding == 'onmf':
            # Find best low dimensional representation.
            q, nfeats, errors = onmf(train_data.X.T)
            W, proj = choose_featureset(
                train_data.X.T, errors, q, nfeats, alpha=3, multiplier=3)

        else:
            W = np.eye(train_data.X.shape[1])
            proj = train_data.X

        # Get control and treated subset of the data and projections.
        idx_control_train = np.where(train_data.obs[
            config.data.condition] == config.data.source)[0]
        idx_treated_train = np.where(train_data.obs[
            config.data.condition] == config.data.target)[0]

        # Compute probabilistic model for control and treated population.
        gmm_control = build_gmm(
            train_data.X[idx_control_train, :].T,
            proj[idx_control_train], ks=(3), niters=2,
            training=.8, criteria='aic')
        gmm_treated = build_gmm(
            train_data.X[idx_treated_train, :].T,
            proj[idx_treated_train], ks=(3), niters=2,
            training=.8, criteria='aic')

        # Compute alignment between components of both mixture models.
        align, _ = align_components(gmm_control, gmm_treated, method="ref2test")

        # Compute perturbation response for each control component.
        res = get_perturbation_response(align, gmm_control, gmm_treated)

        # Save all results to state dict.
        model = {"dim_red": W,
                 "gmm_control": gmm_control,
                 "gmm_treated": gmm_treated,
                 "response": res}
        state_dict = model
        pickle.dump(state_dict, open(cachedir / "last.pt", 'wb'))
        pickle.dump(state_dict, open(cachedir / "model.pt", 'wb'))

    else:
        W = model["dim_red"]
        gmm_control = model["gmm_control"]
        gmm_treated = model["gmm_treated"]
        res = model["response"]

    # Evaluate performance on test set.
    evaluate(config, test_data, model)
