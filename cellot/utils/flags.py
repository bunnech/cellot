from pathlib import Path
import json
from absl import logging, flags
import re

FLAGS = flags.FLAGS


def write_flagfile(path, cfg=None, include_json=False, mode='w', **kwargs):
    '''Pretty prints flagfile from absl.flags

    path: file to write to
    cfg: absl.FLAGS instance
    '''
    if cfg is None:
        cfg = FLAGS

    flag_list_pp = collect_serializable_flags(cfg, **kwargs)
    if hasattr(cfg, 'dry') and cfg.dry:
        logging.info('Dry mode -- will not write %s', str(path))
        return

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Write flagfile to path
    with open(path, mode) as fout:
        sorted_items = sorted(flag_list_pp.items(), key=rank_flag_module)
        for key, srl_list in sorted_items:
            fout.write(f'# {key}\n')
            fout.write('\n'.join(srl_list) + '\n')
            fout.write('\n')

    logging.info('Wrote flags to: %s', str(path))

    if include_json:
        lut = flags_to_json(cfg, **kwargs)
        with open(path.with_name(f'.{path.name}.json'), 'w') as fp:
            json.dump(lut, fp)
    return


def collect_serializable_flags(flags, keyflags=False):
    # Collect all serializable flags
    flag_list_pp = dict()
    flagdict = get_flagdict(flags, keyflags)

    for key, flag_list in flagdict.items():
        flag_list_pp_list = list()
        for fl in flag_list:

            # Only write absl flags if present
            if key.startswith('absl'):
                if not fl.present:
                    continue

            # srl is a flag as it appeared on cmdline
            # e.g. "--arg=value"
            srl = fl.serialize()
            if len(srl) > 1:  # (non-bool) default flags are empty
                flag_list_pp_list.append(srl)

        if len(flag_list_pp_list) > 0:
            flag_list_pp[key] = flag_list_pp_list
    return flag_list_pp


def get_flagdict(cfg, keyflags=False):
    if keyflags:
        return cfg.key_flags_by_module_dict()

    return cfg.flags_by_module_dict()


def rank_flag_module(item):
    '''Sorts tensorflow and absl flags last
    '''
    key, _ = item
    if key.startswith('absl'):
        rank = 100
    elif key.startswith('tensorflow'):
        rank = 10
    else:
        rank = 0
    return (rank, key)


def flags_to_json(cfg, blacklist=None, keyflags=False):
    if blacklist is None:
        blacklist = [r'^tensorflow.*', r'^absl.*']
    blacklist = [re.compile(x) for x in blacklist]
    flagdict = get_flagdict(cfg, keyflags)

    lut = dict()
    for key, flag_list in flagdict.items():
        if key.startswith('tensorflow.') or key.startswith('absl.'):
            continue

        for arg in flag_list:
            if isinstance(arg.value, list):
                val = arg._get_parsed_value_as_string(arg.value)
                val = val.lstrip("'").rstrip("'")

            else:
                val = arg.value

            lut[arg.name] = val

    return lut
