# Runge Kutta 8 Python-OpenCL <br />

Runge Kutta 8 of 13 steps codded with vector4 support. <br />
For a superior number of differential equations vector4 must be replaced, by vector8 or vector16.<br />
Or in any other case vector4 must be replace, by an array. <br />


*Dependencies* <br />
__Numpy__ <br />
__Pyopencl__ <br />
__OpenCL ICD__ <br />


### Execute `runge_kutta` <br />
*For help type `-h` or `--help`*. <br />
*If you have initial conditions in a file, `-A` or `--initial-contions <file_name>`* <br />
*Set initial time `-i` or `--initial-time`* <br />
*Set end time `-f` or `--end-time`* <br />
*Set end time `-d` or `--debug True` initiañ conditions set to 0* <br />
*Set final conditions name file `-S` or `--file_name <file_name>`* <br />
*Print on screen integration step values `-V True` or `--verbose True`. Default = False* <br />
*Print on screen integration step information `-tR True` or `--time-rem True`. Default = True* <br />
For initial conditions from a file change the size of the elements in argument function *read_from_file* <br />



