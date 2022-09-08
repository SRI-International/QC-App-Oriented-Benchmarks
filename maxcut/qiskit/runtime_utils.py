# This script is intended to replace sh_script.sh and script.sed
# The objective of this script is the create the maxcut_runtime.py file
import re
import sys
import os

sys.path[1:1] = [ "_common", "_common/qiskit", "maxcut/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../maxcut/_common/" ]
import common
import execute
import metrics
import maxcut_benchmark

def remove_imports_calls(string):
    pattern_list = [
        "sys.*]\n",
        "import metrics.*\n",
        "import execute as ex\n",
        "import common\n",
        "common\.",
        "ex\.",
        "(?<!_)metrics\.",
        "# print a sample circuit.*\n.*\n",
        "# if main, execute method",
        "if __name__ == '__main__':.*$",
    ]  # remove the printing of sample circuit

    for pattern in pattern_list:
        string = re.sub(pattern, "", string, flags=re.MULTILINE)
    return string


def create_runtime_script(file_name="maxcut_runtime.py"):
    list_of_files = [
        os.path.abspath(common.__file__),
        os.path.abspath(metrics.__file__),
        os.path.abspath(execute.__file__),
        os.path.abspath(maxcut_benchmark.__file__),
    ]
    # Concatenate files
    all_catted_string = ""
    for file in list_of_files:
        with open(file, "r", encoding="utf8") as opened:
            data = opened.read()
        all_catted_string = all_catted_string + "\n" + data

    all_catted_string = remove_imports_calls(all_catted_string)

    # add main file
    with open("add_main.py", "r", encoding="utf8") as text_file:
        data = text_file.read()

    all_catted_string += "\n" + data
    with open(file_name, "w", encoding="utf8") as text_file:
        text_file.write(all_catted_string)

def prepare_instances():
    instance_dir = os.path.join("..", "_common", "instances")
    files = list(filter(lambda x: x.startswith("mc"), os.listdir(instance_dir)))
    insts = {}

    for f in files:
        p = os.path.join(instance_dir, f"{f}")
        k, _, _ = f.partition('.')
        insts[k] = {'instance': None, 'sol': None}

    for f in files:
        p = os.path.join(instance_dir, f"{f}")
        k, _, _ = f.partition('.')
        if 'txt' in f:
            insts[k]['instance'] = common.read_maxcut_instance(p)
        if 'sol' in f:
            insts[k]['sol'] = common.read_maxcut_solution(p)
    
    return insts