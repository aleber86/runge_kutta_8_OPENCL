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
import numpy as np
#import pyopencl as cl
from mod_cmd.module_interface import (escritura_archivo, imp_pantalla,
                                      lectura_archivo, menu_cmd, tiempo_restante,
                                      inicial_time, inicial_date, detencion_programa)
from mod_opencl.mod_opencl_class import Generador


def print_only_not_zero(value):
    suma = np.sum(value[:2])
    if np.abs(suma) > 1.e-20:
        print(value)

def main():
    _paso =np.float64(100) #Paso de integracion
#    global time_lista
    time_lista = []
    tiempo_inicio_prgm = inicial_date()
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
    with open("test.dat", "w") as file_save:
        try:
            depur = [np.float64(0.0) for _ in range(6)]
            inicio = 0
            for valor in range(arg):
                imp_pantalla(valor, arg, valor_i, valor_f, inicio, time_lista)
                Generador_A.ejecucion_kernel(valor_i, valor_f)
                inicio = inicial_time()
                Generador_A.enq_copy()

                index_delete = np.where(((Generador_A.vec_print_3d[:,0]==0.0) &
                                (Generador_A.vec_print_3d[:,1]==0.0)))[0]
                to_file = np.delete(Generador_A.vec_print_3d, index_delete, axis=0)
                np.savetxt(file_save, to_file)
                valor_i = valor_f
                valor_f += _paso
                Generador_A.reset_buffer_print()
                Generador_A.buffer_read(Generador_A.vector_out)
                detencion_programa()
        except StopIteration:
            nombre_detencion = inicial_date()
            escritura_archivo('detencion_'+nombre_detencion+'.dat',
                              Generador_A.vector_out, valor_i, Generador_A._ensambles)
            print(Generador_A.vector_out)
            sys.exit(0)


    print(Generador_A.vector_out)
    escritura_archivo(nombre_arch_res, Generador_A.vector_out, valor_i)
    sys.exit(0)



#------------------------------------------------------------------------------------------------------------
#************************************************************************************************************
#------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
