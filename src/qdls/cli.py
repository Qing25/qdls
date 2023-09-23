# -*- coding: utf-8 -*-
# @File    :   cli.py
# @Time    :   2023/09/23 20:22:19
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''

'''

import os
import sys 
import shutil
from argparse import ArgumentParser


def copy_to_cwd(source):
    """ 
        将source目录下的所有内容复制到cwd目录中
        等价于 cp -r source/* ./
    """
    for root, dirs, files in os.walk(source):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(os.getcwd(), file)
            shutil.copyfile(src_file, dst_file)
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            dst_dir = os.path.join(os.getcwd(), dir)
            shutil.copytree(src_dir, dst_dir)
 

def main():
    """
        在当前目录init
        qinit 
        qinit .
        -----------
        在指定目录init
        qinit new_project
        qinit --dirname new_project
    """
    parser = ArgumentParser()
    parser.add_argument('dirname', nargs="?", type=str, default=None, help='positional argument')
    parser.add_argument('--dirname', dest="opt_dirname", type=str, default=None, help='')
    args = parser.parse_args()
    # 获取模板目录的路径
    template_dir = os.path.join(os.path.dirname(__file__), 'code_template')
    # print(args.dirname, args.opt_dirname)
    # exit()
    if args.opt_dirname is None :
        if args.dirname is None or args.dirname == '.' :
            copy_to_cwd(template_dir)
        else:
            shutil.copytree(template_dir, args.dirname)
    else:
        shutil.copytree(template_dir, args.opt_dirname)

if __name__ == '__main__':
    sys.exit(main())   