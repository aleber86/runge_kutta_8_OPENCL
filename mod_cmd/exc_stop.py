from signal import signal, SIGINT


def signal_handler(sig_rcv, frame):

    print('\nDetenida la integracion por interrupcion de teclado \n')
    raise StopIteration

def detencion_programa():
    signal(SIGINT, signal_handler)
