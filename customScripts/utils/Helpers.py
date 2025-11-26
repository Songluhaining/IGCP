import datetime
import logging
import re
import subprocess
import hashlib
import time
from itertools import chain, combinations

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger(__name__)


def sleep(secs):
    logger.info("Sleeping in {} secs".format(secs))
    time.sleep(secs)


def hash_md5(input_str):
    return hashlib.md5(input_str.encode()).hexdigest()


def hash_to_int(input_str):
    return int(hashlib.sha1(input_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def get_current_timestamp():
    return round(time.time())


def get_current_timestamp():
    return round(time.time())


def get_version_by_time():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')


def find_plugin_name(command):
    matched = re.search("[^ /]+\.(jar|sh)", command)
    if matched:
        return matched.group()


def execute_shell_command(command, extra_args=None, log_to_file=True, show_command=True):
    from FileManager import get_log_file_path, remove_file
    if extra_args:
        for args in extra_args:
            assert isinstance(args, dict), """Input Argument must be passed as dict={"name":"value"}"""
            arg_name = list(args.keys())[0]
            arg_val = args.get(arg_name, None)
            if arg_val is None:
                continue
            command += f" {arg_name} {arg_val}"
    log_file_name = f"{get_current_timestamp()}_{hash_md5(command)}.log"
    log_path = get_log_file_path(log_file_name)
    if show_command:
        print(command)
    with open(log_path, "w+") as outfile:
        plugin_name = find_plugin_name(command)
        if plugin_name and log_to_file:
            logger.debug(f"Writing log [{plugin_name}] to [{log_path}]")
        process = subprocess.run(command, shell=True, stderr=outfile, stdout=outfile)
    with open(log_path, "r") as outfile:
        text = outfile.read()
    if not log_to_file:
        remove_file(log_path)
    return text


def natural_sort(l):
    # in case: sort by file name without considering full path
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key.rsplit("/", 1)[1])]
    return sorted(l, key=alphanum_key)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))