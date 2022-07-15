import os
import yaml
import time
import argparse
from easydict import EasyDict as edict


def parse_arguments():
  parser = argparse.ArgumentParser(description="Running Experiments of GNN")
  parser.add_argument(
      '-c',
      '--config_file',
      type=str,
      default="config/resnet101_cifar.json",
      required=True,
      help="Path of config file")
  parser.add_argument(
      '-l',
      '--log_level',
      type=str,
      default='INFO',
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument('-m', '--comment', help="Experiment comment")
  parser.add_argument('-t', '--test', help="Test model", action='store_true')
  args = parser.parse_args()
  return args


def get_config(config_file, sample_id, exp_dir=None):
  """ Construct and snapshot hyper parameters """
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

  # create hyper parameters
  config.run_id = str(os.getpid())
  if config.model.degree_emb == True:
      config.exp_name = '_'.join([
          config.model.name, "NODE", sample_id, config.dataset.name,
          time.strftime('%Y-%b-%d-%H-%M-%S'), str(config.dataset.split), "SNR_", str(config.ldpc.snrs[0]), str(config.dataset.num_node), config.dataset.data_path.split('/')[-1], str(config.model.hidden_dim), str(config.model.num_prop), config.model.aggregate_type
      ])
  else:
      config.exp_name = '_'.join([config.model.name,
                                  sample_id,
                                  time.strftime('%d-%H-%M'),
                                  str(config.dataset.data_path.split('/')[-1]),
                                  "hidden",
                                  str(config.model.hidden_dim),
                                  "num_prop",
                                  str(config.model.num_prop),
                                  'concat_b' if config.model.include_b else 'not_concat_b',
                                  'lr', str(config.train.lr),
                                  'batch_', str(config.train.batch_size),
                                  "{}module".format(config.model.num_module),
                                  "meta_copy_{}".format(config.train.meta_copy),
                                  "R" if config.train.random_init else "B"
      ])
  if exp_dir is not None:
    config.exp_dir = exp_dir

  config.save_dir = os.path.join(config.exp_dir, config.exp_name)

  # snapshot hyperparameters
  mkdir(config.exp_dir)
  mkdir(config.save_dir)

  save_name = os.path.join(config.save_dir, 'config.yaml')
  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)
