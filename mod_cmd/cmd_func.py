import time
import argparse as agp
from signal import SIGINT, signal


def menu_cmd():
    """Optional arguments for excecution."""

    Arg_parse = agp.ArgumentParser()
    Arg_parse.add_argument('-i','--initial-time',type = float,
                           action='store',
                           help='Set initial time. Default = 0.0\n', default = 0.0)

    Arg_parse.add_argument('-f','--end-time',type = float,
                           action = 'store',
                           help = 'Set end time. Default =  1000 \n',
                           default = 4*10**6)

    Arg_parse.add_argument('-S', '--file-name', type = str,
                           action = 'store',
                           help = 'Set file name for results. Default: "results_/date/time"(.dat)"\n',
                           default = None)

    Arg_parse.add_argument('-A', '--initial-condition', type = str,
                           action = 'store',
                           help = """Uses initial values from file. Needs initial time and end time""",
                           default = None)
    Arg_parse.add_argument('-V', '--verbose', type = bool,
                           action = 'store',
                           help = """Print on screen last integration step. Default = False""",
                           default = False)
    Arg_parse.add_argument('-tR', "--time-rem", type=bool,
                           help = """Print on screen step information and time remaining. Default = True""",
                           action = 'store',
                           default = True)
    value = Arg_parse.parse_args()
    return value


def signal_handler(sig_rcv, frame):

    date = time.strftime('%d/%m/%Y--%H:%M:%S')
    print(f'\nIntegration stopped by keyboard interrupt {date} ','\n')
    raise StopIteration

def execution_kill():
    signal(SIGINT, signal_handler)


def time_remaining(index, total, param, calc_time_size):
    """Estimates end time of integration"""

    time_sec = (sum(param)/calc_time_size) * (total - index)
    hours = time_sec // 3600
    minutes = (time_sec - hours * 3600) // 60
    seconds =  (time_sec - (hours *3600 + minutes * 60))//1
    string_time = f'{hours}h {minutes}m {seconds}s'
    return string_time

def print_on_screen(value, arg, value_i, value_f, time_in_int, param, calc_time_size = 10, active = True ):
    """Print on screen iteration step information."""

    if active:
        print('Iteration index: ',value+1, 'time in: ', value_i,
              'time end: ',value_f, 'total index: ', arg)
        param.append(time.time() - time_in_int)
        if (value+1) % calc_time_size == 0:
            string_rest = time_remaining(value, arg, param, calc_time_size)
            param[:] = []
            print('Estimated end time: ', string_rest)

def time_stamp():
    return time.strftime('%d-%m-%y--%H-%M-%S')

def time_counter():
    return time.time()
