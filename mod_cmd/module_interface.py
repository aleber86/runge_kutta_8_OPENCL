import time
#import subprocess
import argparse as agp
import numpy as np

def menu_cmd():
    #Menu de seleccion de la linea de comandos.
    #Todos los valores estan definidos por defecto
    Arg_parse = agp.ArgumentParser()
    Arg_parse.add_argument('-i','--tiempo-inicial',type = float,
                           action='store',
                           help='Tiempo inicial de integracion. 0.0 por defecto (multiplo de 100)\n', default = 0.0)

    Arg_parse.add_argument('-f','--tiempo-final',type = float,
                           action = 'store',
                           help = 'Tiempo total de la integracion. 4.000.000 por defecto (multiplo de 100)\n',
                           default = 4*10**6)

    Arg_parse.add_argument('-I','--valores-iniciales', type = bool,
                           action = 'store',
                           help = 'Guarda los valores iniciales en un archivo.\n',
                           default = False)

    Arg_parse.add_argument('-n','--nombre-inicial', type = str,
                           action = 'store',
                           help = 'Asigna el nombre del archivo de valores iniciales. Por defecto: "vi_/fecha/hora"(.dat)\n',
                           default = None)

    Arg_parse.add_argument('-S', '--nombre-resultados', type = str,
                           action = 'store',
                           help = 'Asigna el nombre del archivo de resultados. Por defecto: "res_/fecha/hora"(.dat)"\n',
                           default = None)
    Arg_parse.add_argument('-A', '--archivo-iniciales', type = str,
                           action = 'store',
                           help = """Utiliza los valores iniciales del archivo con extension
                                    (tener en cuenta los valores de tiempo inicial y final).""",
                           default = None)

    Arg_parse.add_argument('-V', '--verbose', type = bool,
                           help = 'Imprime por pantalla los valores iniciales y finales en cada integracion\n',
                           action = 'store',
                           default = False)

    Arg_parse.add_argument('-d', '--debug', type = bool,
                           help = 'Genera el conjunto de ensambles con vectores de componentes 1.0. (Debugear el kernel de OpenCL). False por defecto.\n',
                           action = 'store',
                           default = False)

    value = Arg_parse.parse_args()
    return value


def lectura_archivo(archivo, elementos = 768):
    #Funcion de lectura de archivo para los valores
    #iniciales de la integracion en las variables
    #espaciales.
    vector_in = []
    with open(archivo, 'r') as file:
        file.readline()
        file.readline()
        for line in range(elementos):
            arg = file.readline()
            vector_in.append(arg.split())
    vector_in = np.matrix(np.array(vector_in).astype(_wp)).astype(_wp)
    return vector_in


def escritura_archivo(archivo, vector_out, tiempo_f, ensambles = 6):
    #Funcion de escritura de los valores finales de integracion en
    #el intervalo. Debe poseer el mismo formato que la lectura de los
    #valores iniciales para poder hacer uso de cortes intermedios en la
    #integracion.
    with open(archivo, 'w') as file:
        file.write('#Resultados a tiempo: '+ str(tiempo_f)+'\t\t'
                   + 'Cantidad de ensambles: '+ str(ensambles) +'\n')

        file.write('#Theta_1\t\t Theta_2\t\t I_1\t\t I_2\n')
        np.savetxt(file, vector_out, fmt='%.55e', delimiter='  ')
        file.write('\n')
        file.write('\n')

def tiempo_restante(indice, total, param, quant_steps):
   #Calcula el tiempo restante aproximado en funcion del retardo en la integracion
   #del paso actual.
    tiempo_seg = ((sum(param)/quant_steps * (total - indice)))
    horas = tiempo_seg // 3600
    minutos = (tiempo_seg - horas * 3600) // 60
    segundos =  (tiempo_seg - (horas *3600 + minutos * 60))//1
    string_tiempo = f'{horas}h {minutos}m {segundos}s'
    return string_tiempo


def inicial_date():
    return time.strftime('%d-%m-%y--%H-%M-%S')

def inicial_time():
    return time.time()

def imp_pantalla(valor, arg, valor_i, valor_f, tiempo_incio_int,
                 time_lista, quant_steps = 20, activo = True ):
    #Imprime los valores acutales en los pasos de integracion (tiempo, indice, etc)

    if activo:
        print('Indice de integracion: ',valor+1, 'tiempo in: ', valor_i,
              'tiempo fin: ',valor_f, 'indice total: ', arg)
        time_lista.append(time.time() - tiempo_incio_int)
        if (valor+1) % quant_steps == 0:
            #subprocess.call('clear')
            string_rest = tiempo_restante(valor, arg, time_lista, quant_steps)
#            global time_lista
            time_lista[:] = []
            print('Tiempo estimado de finalizcion: ', string_rest)



