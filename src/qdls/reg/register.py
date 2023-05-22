# -*- coding: utf-8 -*-
# @File    :   register.py
# @Time    :   2023/05/22 16:11:26
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    注册器的实现, ref to https://github.com/Delta-ML/delta/blob/master/delta/utils/register.py

    model:                  一些常用的模型实现，默认不加入注册器 only_data=False
    collator:               collator的实现
    process_function:       数据集的输入构建、分词处理
    datamodule:             pytorch lightning 的 datamodule 实现
'''
import importlib
import logging
import os 
import sys 
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

############# 增加新的文件之后需要修改下面的module列表 #############
# 文件夹下面的 py 文件名

MODEL_MODULES = [
    "test_model"
]

COLLATE_MODULES = [
   "common_collators",
]

PROCESS_MODULES = [
   "kqa"
]

DM_MODULES = [
    "base_dm"
]

ALL_DATA_MODULES = [
    ("qdls.reg.collators", COLLATE_MODULES),
    ("qdls.reg.process_functions", PROCESS_MODULES),
    ("qdls.reg.datamodules", DM_MODULES),
]

ALL_MODULES = [
   ("qdls.reg.models", MODEL_MODULES), 
] + ALL_DATA_MODULES

########## 以下内容不需要修改 ##########
 
class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        """ 取函数名作为key """
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def get(self, key):
        """ get a data process function from registry"""
        return self._dict[key]


class registers:
    model = Register('model')
    collator = Register('collator')
    process_function = Register('process_function')
    datamodule = Register('datamodule')
    
    def __init__(self) -> None:
        raise RuntimeError("Registries is not intended to be instantiated")
    

def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  for name, err in errors:
    logging.warning("Module {} import failed: {}".format(name, err))
  logging.fatal("Please check these modules.")


def path_to_module_format(py_path):
  """Transform a python file path to module format."""
  return py_path.replace("/", ".").rstrip(".py")


def add_custom_modules(all_modules, config=None):
  """Add custom modules to all_modules"""
  current_work_dir = os.getcwd()
  if current_work_dir not in sys.path:
    sys.path.append(current_work_dir)
  if config is not None and "custom_modules" in config:
    custom_modules = config["custom_modules"]
    if not isinstance(custom_modules, list):
      custom_modules = [custom_modules]
    all_modules += [
        ("", [path_to_module_format(module)]) for module in custom_modules
    ]


def import_all_modules_for_register(config=None, only_data=True):
  """Import all modules for register."""
  if only_data:
    all_modules = ALL_DATA_MODULES
  else:
    all_modules = ALL_MODULES

  add_custom_modules(all_modules, config)

  logging.debug(f"All modules: {all_modules}")
  errors = []
  for base_dir, modules in all_modules:
    for name in modules:
      try:
        if base_dir != "":
          full_name = base_dir + "." + name
        else:
          full_name = name
        # logging.warning(f"Importing {full_name}...")
        importlib.import_module(full_name)
        logging.debug(f"{full_name} loaded.")
      except ImportError as error:
        errors.append((name, error))
  _handle_errors(errors)


