# This script is intended to replace sh_script.sh and script.sed
# The objective of this script is the create the maxcut_runtime.py file
import re
import sys
import os
import json
from collections import defaultdict 

sys.path[1:1] = [ "_common", "_common/qiskit", "maxcut/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../maxcut/_common/" ]
import common
import execute
import metrics
import maxcut_benchmark

from qiskit_ibm_runtime import QiskitRuntimeService


def remove_imports_calls(string):
    pattern_list = [
        "sys.*]\n",
        "import metrics.*\n",
        "import execute as ex\n",
        "import common\n",
        "common\.",
        "ex\.",
        "(?<!_)metrics\.",
        # "# print a sample circuit.*\n.*\n"
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
    insts = defaultdict(dict)
    instance_dir = os.path.join("..", "_common", "instances")
    files = (file for file in os.listdir(instance_dir) if file.startswith("mc"))

    for f in files:
        p = os.path.join(instance_dir, f"{f}")
        k, _, _ = f.partition('.')
        if 'txt' in f:
            insts[k]['instance'] = common.read_maxcut_instance(p)
        if 'sol' in f:
            insts[k]['sol'] = common.read_maxcut_solution(p)
            
    common_dir = os.path.join("..", "_common")
    insts['fixed_angles'] = common.read_fixed_angles(
            os.path.join(common_dir, 'angles_regular_graphs.json'))
    
    return insts


def get_status(service, job_id):
    return service.job(job_id=job_id).status().name

def save_jobinfo(backend_id, job_id, job_status):
    path = os.path.join("__data", backend_id)
    line = f"{job_id},{job_status}"

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "jobs.txt"), "w+") as file:
        file.write(line)
 
''' 
def save_jobinfo(backend_id, job_id, job_status):
    path = os.path.join("__data", backend_id)
    line = f"{job_id},{job_status}\n"

    os.makedirs(path, exist_ok=True)
    try:
        with open(os.path.join(path, "jobs.txt"), "r+") as file:
            data = file.readlines()
    except FileNotFoundError:
        with open(os.path.join(path, "jobs.txt"), "w+") as file:
            data = file.readlines()

    if line in data:
        return

    with open(os.path.join(path, "jobs.txt"), "w+") as file:
        if job_status == "DONE":
            data[-1] = data[-1].replace("RUNNING", job_status)
        else:
            data.append(line)

        file.write("".join(data))
'''

def get_jobinfo(backend_id):
    path = os.path.join("__data", backend_id, "jobs.txt")
    try:
        with open(path, "r") as file:
            data = file.read()
    except FileNotFoundError:
        return [None, None]
    
    job_id, status = data.strip().split(",")
    job = QiskitRuntimeService().job(job_id=job_id)
    return job, status


def get_id(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as file:
        data = file.read()

    job_id, status = data.split(",")

    return job_id, status


def get_response(service, path):
    if not os.path.exists(path):
        return "continue"

    job_id = get_id(path)
    status = get_status(service, job_id)
    print(
        f"WARNING: Job file already exists! Job {job_id} is {status}"
    )
    response = input(
        f"Would you like to continue and overwrite your previous data in {os.path.dirname(path)}? (y/n)"
    )

    return response


def process_results(job_id, backend_id, service):
    path = os.path.join("__data", f"{backend_id}")
    # Will wait for job to finish
    result = service.job(job_id).result()
    maxcut_benchmark.save_runtime_data(result)
    maxcut_benchmark.load_data_and_plot(path)


def run(**kwargs):
    service = QiskitRuntimeService()

    options = {
        'backend_name': kwargs['backend_id']
    }

    runtime_inputs = {
        "backend_id": kwargs['backend_id'],
        "method": 2,
        "_instances": kwargs['_instances'],
        "min_qubits": kwargs['min_qubits'],
        "max_qubits": kwargs['max_qubits'],
        "max_circuits": kwargs['max_circuits'],
        "num_shots": kwargs['num_shots'],

        "degree": kwargs['degree'],
        "rounds": kwargs['rounds'],
        "max_iter": kwargs['max_iter'],
        "parameterized": kwargs['parameterized'],
        "do_fidelities": False,

        # To keep plots consistent
        "hub": kwargs['hub'],
        "group": kwargs['group'],
        "project": kwargs['project']
    }

    job_file_path = os.path.join(
        "__data", f"{kwargs['backend_id']}", "job.txt"
    )
    if os.path.exists(job_file_path):
        response = get_response(service, job_file_path)
        job_id = get_id(job_file_path)

        if response.strip().lower() == "n":
            print("Aborting without executing any procedures.")
            return

        status = get_status(service, job_id)
        if status != 'ERROR' or status != 'CANCELLED':
            print("Fetching previously submitted job:")
            process_results(job_id, kwargs["backend_id"], service)
            os.remove(job_file_path)
            return 

    RUNTIME_FILENAME = 'maxcut_runtime.py'
    create_runtime_script(file_name=RUNTIME_FILENAME)
    program_id = service.upload_program(
        data=RUNTIME_FILENAME, metadata=kwargs["meta"]
    )
    ## Uses previously uploaded program instead of uploading a new one.
    # program_id = list(
    #     filter(
    #         lambda x: x.program_id.startswith("qedc"), 
    #         service.programs()
    #     )
    # )[0].program_id

    job = service.run(
        program_id=program_id,
        options=options,
        inputs=runtime_inputs,
        instance=f'{kwargs["hub"]}/{kwargs["group"]}/{kwargs["project"]}'
    )

    try:
        with open(job_file_path, "w+") as file:
            file.write(f"{job.job_id},{job.status().name}")
    except FileNotFoundError:
        os.mkdir(os.path.join('__data', f"{kwargs['backend_id']}"))
        with open(job_file_path, "w+") as file:
            file.write(f"{job.job_id},{job.status().name}")

    process_results(job.job_id, kwargs["backend_id"], service)

    return job