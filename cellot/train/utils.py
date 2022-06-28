import torch
from pathlib import Path
from absl import logging
from cellot.utils.helpers import flat_dict, nest_dict


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
        logging.warn(f'\'{key}\' not found in ckpt: {str(path)}')
        return default

    return ckpt[key]


def cast_loader_to_iterator(loader, cycle_all=True):
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    iterator = nest_dict({
        key: cycle(item)
        for key, item
        in flat_dict(loader).items()
    }, as_dot_dict=True)

    return iterator
