import sys
from collections import namedtuple

import yaml
from absl import flags

import cellot.train
from cellot.train.experiment import prepare
from cellot.utils.helpers import symlink_to_logfile, write_metadata, flat_dict
import wandb


Pair = namedtuple("Pair", "source target")

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config", "", "Path to config")

flags.DEFINE_string("exp_group", "cellot_exps", "Name of experiment.")

flags.DEFINE_string("online", "offline", "Run experiment online or offline.")

flags.DEFINE_boolean("restart", False, "delete cache")
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_boolean("dry", False, "dry mode")
flags.DEFINE_boolean("verbose", False, "run in verbose mode")


def main(argv):
    config, outdir = prepare(argv)
    if FLAGS.dry:
        print(outdir)
        print(config)
        return

    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)

    yaml.dump(
        config.to_dict(), open(outdir / "config.yaml", "w"), default_flow_style=False
    )

    symlink_to_logfile(outdir / "log")
    write_metadata(outdir / "metadata.json", argv)

    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    if FLAGS.restart:
        (cachedir / "model.pt").unlink(missing_ok=True)
        (cachedir / "scalars").unlink(missing_ok=True)

    wandb_config = flat_dict(config.to_dict())

    # init wandb
    wandb.init(
        project="exp_single",
        dir=cachedir,
        group=FLAGS.exp_group,
        entity="teamcot",
        config=wandb_config,
        mode=FLAGS.online,
    )

    if config.model.name == "cellot":
        train = cellot.train.train_cellot

    elif config.model.name == "scgen" or config.model.name == "cae":
        train = cellot.train.train_auto_encoder

    elif config.model.name == "identity":
        return

    elif config.model.name == "random":
        return

    else:
        raise ValueError

    # start training
    status = cachedir / "status"
    status.write_text("running")

    try:
        train(outdir, config)
    except ValueError as error:
        status.write_text("bugged")
        print("Training bugged")
        raise error
    else:
        status.write_text("done")
        print("Training finished")

    return


if __name__ == "__main__":
    main(sys.argv)
