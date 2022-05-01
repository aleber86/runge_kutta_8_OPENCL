#Programa para la integracion de las ecuaciones de movimiento
#del modelo de Arnold de hamiltoniano perturbado en dos dimensiones,
#en resonancia multilineal.
#Se busca obtener los mismos resultados que lo expuesto en el trabajo
#para la difusion en tiempos de T_a = 4*10**6, obrante en
#On the chaotic diffusion in multidimensional Hamiltonian Systems
#(Cincotta, Giordano, Marti y Beague) en el menor
#tiempo posible. En funcion de esto, se programo un kernel para OpenCL con el objetivo
#de ser ejecutado en paralelo en una GPGPU (tambien en CPU multinucleo)

#El trabajo al que se hace referencia toma 6 ensambles de 100 valores con condiciones
#iniciales para las variables angulo-accion (theta_1, I_1, theta_2, I_2) -> (pi, 0.0, 1.0(var), I_2(var))
#Siendo los centros de los ensambles para la frec. angular_2 [-1.69, -1.13, -0.173, 0.053, 0.55, 1.55]
#tiempo inicial 0.
#donde (var) representa variaciones en las condiciones iniciales para las 100 muestras
#por ensamble en el orden de 10**-7.

#En funcion de la geometria del dispositivo a utilizar:
#GPGPU AMD Pitcairn HD 7850 16 nucleos 860 MHz, 1.4 GB VRAM DDR5 bus de memoria en 256 bits 1200 MHz,
#el cual hace mejor uso de sus capacidades en dimensiones locales totales multiplo de 64;
#se tomaron 6 ensambles, pero de 128 elementos cada una, respetando las condiciones iniciales,
#haciendolas variar en forma pseudoaleatoria con el orden de mangniutd propuesto en el trabajo precendete.

import sys
import subprocess
from signal import signal, SIGINT
import time
import numpy as np
import pyopencl as cl
import argparse as agp

#######################################################
#Variables de entorno globales
#######################################################
global _pi, _wp, _vec_comp, _paso
_pi = np.float64(np.pi) #Def. PI en numpy
_wp = np.float64  #REAL*8
_bytes_sz = np.float64(0).nbytes
_vec_comp = np.int64(4) #Cuatro componentes espaciales (vectores)
_paso =np.float64(100) #Paso de integracion
#######################################################



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

def tiempo_restante(indice, total, param):
   #Calcula el tiempo restante aproximado en funcion del retardo en la integracion
   #del paso actual.
    tiempo_seg = ((sum(param)/100 * (total - indice)))
    horas = tiempo_seg // 3600
    minutos = (tiempo_seg - horas * 3600) // 60
    segundos =  (tiempo_seg - (horas *3600 + minutos * 60))//1
    string_tiempo = f'{horas}h {minutos}m {segundos}s'
    return string_tiempo

def imp_pantalla(valor, arg, valor_i, valor_f, tiempo_incio_int, param, activo = True ):
    #Imprime los valores acutales en los pasos de integracion (tiempo, indice, etc)

    if activo:
        print('Indice de integracion: ',valor+1, 'tiempo in: ', valor_i,
              'tiempo fin: ',valor_f, 'indice total: ', arg)
        param.append(time.time() - tiempo_incio_int)
        if (valor+1) % 100 == 0:
            #subprocess.call('clear')
            string_rest = tiempo_restante(valor, arg, param)
            global time_lista
            time_lista = []
            print('Tiempo estimado de finalizcion: ', string_rest)


#----------------------------------------------------------------------------------------------------
#BLOQUE DE CLASE PARA LA EJECUCION DEL KERNEL
#----------------------------------------------------------------------------------------------------
class Generador():
    def __init__(self, dim_vec_x_ens =128 , t_int =10**4, lsz = 64, debug = False):
        """
        La clase Generador define todos los parametros a utilizar por la API de OpenCL.
        Se espera que no se realicen modificaciones ya que el problema propuesto esta
        determinado. En caso de modificar los valores dimensionales de los vectores (4 -> X),
        se debe modificar el kernel adjunto como RK8_11.cl
        """
        self.DEBUG = debug
        self._dim = np.int32(dim_vec_x_ens)
        self._ensambles = np.int32(6)
        self.dim_print = np.int32(5000)
        self._dim_tiempo_int = np.int64(t_int) #Siempre tiene que ser un np.int64(int)
        self.vector_in = []
        self.vector_out = []
        self.vec_print_3d = []
        self.exit_c = None
        self.index_vector_print = None
        ####################
        #Variables de OpenCL
        ####################
        self.contexto = None
        self.queue = None
        self.mem_flag = None
        self.vec_out_dev = None
        self.vec_in_dev = None
        self.kernel = None
        self.local_mem = None
        self.local_mem_k = None
        self.local_mem_y = None
        self.program = None
        self.exit_d = None
        self.local_sz = lsz
        self.vec_print_3d_dev = None
        self.index_vector_print_dev = None

    def valores_iniciales(self, cond_in = [np.float64(-1.69), np.float64(-1.13),
                                           np.float64(-0.173),np.float64( 0.053),
                                           np.float64(0.55),np.float64(1.55)]):
        #Los valores de la lista corresponden a los centros de los ensambles en
        #frecuencia angualar_2

        self._ensambles = len(cond_in) #Cntidad de ensambles en funcion de las cond_in.
        self.vector_in = np.array([]).astype(_wp)

        for val_in in cond_in:
            if self.DEBUG:
                #Opcion debug=True (cond. in = 1 p/toda componente y vector)
                vector_fab = np.array([[np.float64(1.0),np.float64(1.0),
                                        np.float64(1.0), np.float64(1.0)]
                                       for k in range(self._dim)]).astype(_wp)
            else:
                #Opcion debug=False (cond. in normales del problema)
                vector_fab = np.array([[np.float64(_pi), np.float64(0.0),
                                        np.float64(1.0) +
                                        np.float64((np.float64(-1))**j*((np.float64(1.0)-np.random.random())*np.float64(10**-6))),
                                       val_in + np.float64((np.float64(-1))**j*((1.0 - np.random.random())*np.float64(10**-6)))]
                                       for j in range(self._dim)]).astype(_wp)
            if len(self.vector_in) == 0:
                self.vector_in = vector_fab
            else:
                self.vector_in = np.vstack((self.vector_in, vector_fab))
        self.vector_in = np.matrix(self.vector_in).astype(_wp) #Forma matricial de los valores iniciales.

        self.vector_out_asign()
        del vector_fab #Se elimina la matriz auxiliar para reducir uso de memoria en ejecucion.

        self.reset_buffer_print()

    def vector_out_asign(self):
        self.vector_out = np.empty_like(self.vector_in).astype(_wp)

    def OpenCL_contexto(self, archivo = 'gpu.cl'):
        #Se genera el contexto como los buffer de memoria
        #Se establece el tamano de la memoria local


        self.contexto = cl.create_some_context()
        self.queue = cl.CommandQueue(self.contexto)
        self.mem_flag = cl.mem_flags
        self.buffer_write()
        self.buffer_read()
        with open(archivo, 'r') as file_kernel:
            kernel_script = file_kernel.read()


        self.program = cl.Program(self.contexto, kernel_script).build()
        ####################################################################
        ####################################################################

        print(cl.program_build_info)
        self.kernel = self.program.Arnold_H #Programa dentro del kernel de OpenCL

        #8 bytes por float64 son 128*6*4 (vec, ensa, comp. vec)---Espacio en memoria local_mem
        #del dispositivo.
        self.local_mem = cl.LocalMemory(self.local_sz*(5)*_bytes_sz)  #VER ejecucion_kernel() tamano mem_local
        self.local_mem_y = cl.LocalMemory(self.local_sz*np.int32(0).nbytes)
        self.local_mem_k = cl.LocalMemory(np.int32(0).nbytes)
    def buffer_write(self, arg = None):
        #Construye el buffer de salida; se espara un solo uso, pero se programa para el caso que se quiera
        #ejecutar otro kernel escrito y re-usar el buffer de memoria junto con el queue y el contexto

        #self.exit_c genera un conjunto de 1 por cada vector dentro de cada ensamble.
        #Control de integracion. Ver dopri.cl, error de integracion -> salida 1 en el vector
        #mal integrado.

        self.exit_c = np.asarray([[np.int32(0) for i in range(self._dim)]
                                  for j in range(self._ensambles)]).astype(np.int32)
        if arg is None:
            arg = self.vector_out
        self.exit_d = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                self.mem_flag.COPY_HOST_PTR, hostbuf = self.exit_c)

        self.vec_out_dev = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                     self.mem_flag.COPY_HOST_PTR, hostbuf = arg)

        self.vec_print_3d_dev = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                          self.mem_flag.COPY_HOST_PTR, hostbuf = self.vec_print_3d)
    def buffer_read(self, arg = None, arg2 = None, arg3 = None):
        #Construye el buffer de entrada y se usa para la re-entrada de datos procesados
        if arg is None:
            arg = self.vector_in

        if arg2 is None:
            arg2 = self.vec_print_3d

        if arg3 is None:
            arg3 = self.index_vector_print

        self.vec_in_dev =  cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                     self.mem_flag.COPY_HOST_PTR, hostbuf = arg)
        self.vec_print_3d_dev =  cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                           self.mem_flag.COPY_HOST_PTR, hostbuf = arg2)
        self.index_vector_print_dev = cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                                self.mem_flag.COPY_HOST_PTR, hostbuf = arg3)

    def enq_copy(self, ):
        #Copia los valores procesados por el dispositivo a la memoria del host
        #la funcion se ejecuta cada vez que se va a reiniciar un ciclo de calculo.

        cl.enqueue_copy(self.queue, self.vector_out, self.vec_out_dev )
        cl.enqueue_copy(self.queue, self.exit_c, self.exit_d)
        cl.enqueue_copy(self.queue, self.vec_print_3d, self.vec_print_3d_dev)
        cl.enqueue_copy(self.queue, self.index_vector_print, self.index_vector_print_dev)
        self.vector_out.astype(_wp)


    def ejecucion_kernel(self, t_i = 0, t_f = 100):
        #Ejecucion del kernel.
        #Los valores iniciales pueden ser cambiados por argumentos desde la linea
        #de comandos.

        tiempo_i = np.float64(t_i)
        tiempo_f = np.float64(t_f)
        self.kernel(self.queue, (self._dim*self._ensambles,), (self.local_sz,),self.index_vector_print_dev,self._dim,
                    self.dim_print, self.vec_out_dev, self.vec_in_dev, self._dim_tiempo_int, tiempo_i, tiempo_f,
                    self.local_mem, self.vec_print_3d_dev, self.exit_d, self.local_mem_k, self.local_mem_y)

    def reset_buffer_print(self):
        self.vec_print_3d = np.zeros((self._ensambles,self._dim*self.dim_print,5)).astype(_wp)
        self.index_vector_print = np.zeros((self._ensambles,self._dim)).astype(np.int32)
#------------------------------------------------------------------------------------------------------------
#************************************************************************************************************
#------------------------------------------------------------------------------------------------------------
def signal_handler(sig_rcv, frame):

    fecha = time.strftime('%d/%m/%Y--%H:%M:%S')
    print('\nDetenida la integracion por interrupcion de teclado: ', fecha,'\n')
    raise StopIteration

def detencion_programa():
    signal(SIGINT, signal_handler)



if __name__ == '__main__':
    global time_lista
    time_lista = []
    tiempo_inicio_prgm = time.strftime('%d-%m-%y--%H-%M-%S')
    #------------------------------------
    #Lineas para argparse

    argumentos = menu_cmd()
    valor_i = argumentos.tiempo_inicial
    tiempo_f_total = argumentos.tiempo_final
    if (valor_i >= tiempo_f_total) or (valor_i%100 != 0 or tiempo_f_total%100 !=0 ):
        print("""\n***ERROR:\n
#Los valores para tiempo inicial deben \
#ser menores que los de tiempo final total y multiplos de 100""")
        sys.exit(-1)

    nombre_arch_inicial = 'vi_'+tiempo_inicio_prgm+ '.dat'
    nombre_arch_res = 'res_'+tiempo_inicio_prgm+ '.dat'
    if argumentos.nombre_inicial is not None:
        nombre_arch_inicial = argumentos.nombre_inicial
    if argumentos.nombre_resultados is not None:
        nombre_arch_res = argumentos.nombre_resultados

    #-------------------------------------------------------------------------------
    #Se generan los valores para el ciclo de calculo
    valor_f = np.float64(valor_i) + _paso
    #valor_f = tiempo_f_total
    arg =int((np.float64(tiempo_f_total) - valor_i) / _paso) #Se debe modificar con argparse
    #--------------------------------------------------------------------------------
    Generador_A = Generador(debug = argumentos.debug)

    if argumentos.archivo_iniciales is not None and not Generador_A.DEBUG:
        #Se utilizan los valores iniciales almacenados en el archivo
        #siempre que no se vaya a utilizar el debug

        arch_a_utilizar = argumentos.archivo_iniciales
        Generador_A.vector_in = lectura_archivo(arch_a_utilizar)
        Generador_A.vector_out_asign()
    else:
        #De cualquier otra manera, se utiliza la funcion intrinseca generadora
        #de valores iniciales.

        Generador_A.valores_iniciales()

    if argumentos.valores_iniciales:
        escritura_archivo(nombre_arch_inicial, Generador_A.vector_in, Generador_A._ensambles)

    Generador_A.OpenCL_contexto()
    try:
        inicio = 0
        for valor in range(arg):
            imp_pantalla(valor, arg, valor_i, valor_f, inicio, time_lista)
            Generador_A.ejecucion_kernel(valor_i, valor_f)
            inicio = time.time()
            Generador_A.enq_copy()
            print(Generador_A.index_vector_print)
            valor_i = valor_f
            valor_f += _paso
            Generador_A.reset_buffer_print()
            Generador_A.buffer_read(Generador_A.vector_out)
            detencion_programa()
    except StopIteration:
        nombre_detencion = time.strftime('%d-%m-%y--%H-%M-%S')
        escritura_archivo('detencion_'+nombre_detencion+'.dat',
                          Generador_A.vector_out, valor_i, Generador_A._ensambles)
        print(Generador_A.vector_out)
        sys.exit(0)
    except StopAsyncIteration:
        with open('log_error.log','w') as log:
            log.write('***Error en la integracion\n')
            log.write('Tiempo: '+ str(valor_f) + '\n')
            np.savetxt(log, lista_error, fmt='%1i', delimiter = '  ')
            print(Generador_A.vector_out)
        escritura_archivo(nombre_arch_res, Generador_A.vector_out, valor_f)
        sys.exit(-1)


    print(Generador_A.vector_out)
    escritura_archivo(nombre_arch_res, Generador_A.vector_out, valor_i)
    sys.exit(0)

