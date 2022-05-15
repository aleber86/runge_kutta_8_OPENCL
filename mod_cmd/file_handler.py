from numpy import (float64 as np_float64,
                   matrix as np_matrix,
                   savetxt as np_savetxt,
                   array as np_array)

def read_from_file(file_name, elements = 768, _wp = np_float64):
    """Reading from drive function. Takes the elements in last
    iteration or sets new initial conditions"""

    vector_in = []
    with open(file_name, 'r') as file:
        file.readline()
        file.readline()
        for line in range(elements):
            arg = file.readline()
            vector_in.append(arg.split())
    vector_in = np_matrix(np_array(vector_in), dtype=_wp)
    return vector_in


def write_to_file(file_name, vector_out, time_end, ensambles = 6):
    """Writes to a file last iteration: time and values"""

    with open(file_name, 'w') as file:
        file.write('#Results at time: '+ str(time_end)+'\t\t'
                   + 'ensambles: '+ str(ensambles) +'\n')

        file.write('#Arg_1\t\t Arg_2\t\t Arg_3\t\t Arg_4\n')
        np_savetxt(file, vector_out, delimiter='  ')
        file.write('\n')
        file.write('\n')
