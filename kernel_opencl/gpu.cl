#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64
#include"coeficient_block.h"

double4 diff_eq(double4 arg, double time_value){
    double4 result;
    result.x = time_value;
    result.y = arg[0];
    result.z = arg[1];
    result.w = arg[2];
    return result;
}
    
    
    void integration_function(double t, double4 *y_b, double tfin, double epsilon,
        double hmax, double h, 
         __global int *exi, int echo, int gid){


    double4 y11s, y12s, k[7], y_l, y;
    bool reject, ok, ok2;
    int nfcn, nstep, naccpt, nreject; 
    double xph, fac, hnew;
    double uround, posneg;
    double err, denom;
    int nmax;

    y = *y_b;

    /*  double cOEF:
     *  ************/
  
     /* 
     double c2 = 0.55555555555555552471602709374565E-01;
     double c3 = 0.83333333333333328707404064061848E-01;
     double c4 = 0.12500000000000000000000000000000;    
     double c5 = 0.31250000000000000000000000000000;    
     double c6 = 0.37500000000000000000000000000000;    
     double c7 = 0.14749999999999999222843882762390;    
     double c8 = 0.46500000000000002442490654175344;
     double c9 = 0.56486545138225952022992260026513;    
     double c10 = 0.65000000000000002220446049250313;    
     double c11 = 0.92465627764050439818532822755515;
    //c12 = 1.0000000000000000000000000000000; No se usan    
    //c13 = 1.0000000000000000000000000000000;    
     double a21 = 0.55555555555555552471602709374565E-01;
     double a31 = 0.20833333333333332176851016015462E-01;
     double a32 = 0.62500000000000000000000000000000E-01;
     double a41 = 0.31250000000000000000000000000000E-01;
     double a43 = 0.93750000000000000000000000000000E-01;
     double a51 = 0.31250000000000000000000000000000;    
     double a53 = -1.1718750000000000000000000000000;    
     double a54 = 1.1718750000000000000000000000000;
     double a61 = 0.37499999999999998612221219218554E-01;
     double a64 = 0.18750000000000000000000000000000;
     double a65 = 0.14999999999999999444888487687422;    
     double a71 = 0.47910137111111111840600074174290E-01;
     double a74 = 0.11224871277777777323070296233709;    
     double a75 = -0.25505673777777779220876652743755E-01;
     double a76 = 0.12846823888888888112735919833085E-01;
     double a81 = 0.16917989787292281311792407905159E-01;
     double a84 = 0.38784827848604319644465476812911;    
     double a85 = 0.35977369851500330677485095520751E-01;
     double a86 = 0.19697021421566607290998263124493;    
     double a87 = -0.17271385234050184998011445713928;    
     double a91 = 0.69095753359192296771951191658445E-01;
     double a94 = -0.63424797672885413479804128655815;    
     double a95 = -0.16119757522460406717890180061659;    
     double a96 = 0.13865030945882525492685033441376;    
     double a97 = 0.94092861403575622780692810920300;    
     double a98 = 0.21163632648194397045671166779357;    
     double a101 = 0.18355699683904538876966228144738;    
     double a104 = -2.4687680843155925813903195376042;    
     double a105 = -0.29128688781630046600312766713614;    
     double a106 = -0.26473020233117375982212493568113E-01;
     double a107 = 2.8478387641928004647695615858538;    
     double a108 = 0.28138733146984978850113634507579;    
     double a109 = 0.12374489986331466129243494833645;    
     double a111 = -1.2154248173958881462652925620205;    
     double a114 = 16.672608665945773509520222432911;    
     double a115 = 0.91574182841681794897681356815156;    
     double a116 = -6.0566058043574706459821754833683;    
     double a117 = -16.003573594156179638048342894763;    
     double a118 = 14.849303086297663156756243552081;    
     double a119 = -13.371575735289848552156399819069;    
     double a1110 = 5.1341826481796379866295865213033;    
     double a121 = 0.25886091643826425467977969674394;    
     double a124 = -4.7744857854892046589156961999834;    
     double a125 = -0.43509301377703252233786201941257;    
     double a126 = -3.0494833320722416480919036985142;    
     double a127 = 5.5779200399360995277220354182646;    
     double a128 = 6.1558315898610400651591589848977;    
     double a129 = -5.0621045867369387494250076997560;    
     double a1210 = 2.1939261731806789512688737886492;    
     double a1211 = 0.13462799865933494647407542288420;    
     double a131 = 0.82242759962650746619061692399555;    
     double a134 = -11.658673257277664347952850221191;    
     double a135 = -0.75762211669093615373782313326956;    
     double a136 = 0.71397358815958156252889921233873;    
     double a137 = 12.075774986890056794663905748166;    
     double a138 = -2.1276591139204028557685433042934;    
     double a139 = 1.9901662070489554157148859303561;    
     double a1310 = -0.23428647154404028118968028593372;
     double a1311 = 0.17589857770794226077271105168620;    
     double b1 = 0.41747491141530243541346578695084E-01;
     double b6 = -0.55452328611239311284553110681372E-01;
     double b7 = 0.23931280720118008886743155017029; 
     double b8 = 0.70351066940344297861997802101541;
     double b9 = -0.75975961381446088793722992704716;
     double b10 = 0.66056303092228629836313302803319;
     double b11 = 0.15818748251012332284304306995182;    
     double b12 = -0.23810953875286280934098215311678;    
     double b13 = 0.25000000000000000000000000000000;    
     double bh1 = 0.29553213676353499300697436069640E-01;
    double bh6 = -0.82860627648779705545223350782180;    
    double bh7 = 0.31124090005111831880313388865034;    
    double bh8 = 2.4673451905998868838310045248363;    
    double bh9 = -2.5469416518419087935853895032778;    
    double bh10 = 1.4435485836767751877118826087099;    
    double bh11 = 0.79415595881127287736234166004579E-01;
    double bh12 = 0.44444444444444446140618509843989E-01;
*/

    nmax = 6000;   //Maximum value of iteration.
    uround = 1.73e-18; // 1.e0+uround>1.e0

    posneg = copysign(1.e0, tfin - t);
    hmax = fabs(hmax);
    h = min(max(1.e-10, fabs(h)), hmax);
    h = copysign(h, posneg);
    epsilon = max(epsilon, 13.e0 * uround);
    reject = false;
    ok = true;
    naccpt = 0;
    nreject = 0;
    nfcn = 0;
    nstep = 0;
    

    while(ok){
    ok2 = true;
    if (nstep > nmax || t+0.3e0*h == t){
        exi[gid] = 1;
       echo = 1;
        return ;
    }

    if((t-tfin)*posneg+uround > 0.e0){
        
        return ;
    }
    if ((t+h-tfin)*posneg > 0.e0){
        h = tfin-t;
    }
    //K1:
    k[0] = diff_eq(y, t);
    //double adouble cA VIENE UN WHILE PARA CORREGIR EL GOTO
    while(ok2){
    nstep++;

    //9 pasos......

    y_l = y + h*a21*k[0];

    //K2:
    k[1] = diff_eq(y_l, t+c2*h);
    y_l = y + h*(a31*k[0] + a32*k[1]);

    //K3:
    k[2] = diff_eq(y_l, t+c3*h);
    y_l = y + h*(a41*k[0] + a43*k[2]);

    //K4:

    k[3] = diff_eq(y_l, t+c4*h);
    y_l = y + h*(a51*k[0] + a53*k[2] + a54*k[3]);


    //K5:

    k[4] = diff_eq(y_l, t+c5*h);
    y_l = y + h*(a61*k[0] + a64*k[3] +a65*k[4]);

    //K6:

    k[5] = diff_eq(y_l, t+c6*h);
    y_l = y + h*(a71*k[0] + a74*k[3] + a75*k[4] +
            a76*k[5]);

    //K7:

    k[6] = diff_eq(y_l, t+c7*h);
    y_l = y + h*(a81*k[0] + a84*k[3] + a85*k[4] +
            a86*k[5] + a87*k[6]);

    //K8:
    
    k[1] = diff_eq(y_l, t+c8*h);
    y_l = y + h*(a91*k[0] + a94*k[3] + a95*k[4] +
            a96*k[5] + a97*k[6] + a98*k[1]);

    //K9:

    k[2] = diff_eq(y_l, t+c9*h);
    y_l = y + h*(a101*k[0] + a104*k[3] + a105*k[4] +
            a106*k[5] + a107*k[6] + a108*k[1] + a109*k[2]);

    y11s = a111*k[0] + a114*k[3] + a115*k[4] + a116*k[5] + 
        a117*k[6] + a118*k[1] + a119*k[2];

    y12s = a121*k[0] + a124*k[3] + a125*k[4] + a126*k[5] + 
        a127*k[6] + a128*k[1] + a129*k[2];

    k[3] = a131*k[0] + a134*k[3] + a135*k[4] + a136*k[5] + 
        a137*k[6] + a138*k[1] + a139*k[2];

    k[4] = b1*k[0] + b6*k[5] + b7*k[6] + b8*k[1] + b9*k[2];
    k[5] = bh1*k[0] + bh6*k[5] + bh7*k[6] + bh8*k[1] + 
        bh9*k[2];
    k[1] = y11s;
    k[2] = y12s;

    //Las 4 etapas faltantes.

    k[6] = diff_eq(y_l, t+c10*h);
    y_l = y + h*(k[1] + a1110*k[6]);
    k[1] = diff_eq(y_l, t+c11*h);
    xph = t+h;

    y_l = y + h*(k[2] + a1210*k[6] + a1211*k[1]);

    k[2] = diff_eq(y_l, xph);

    y_l = y + h*(k[3] + a1310*k[6] + a1311*k[1]);

    k[3] = diff_eq(y_l, xph);

    nfcn = nfcn + 13;

    k[4] = y + h*(k[4] + b10*k[6] + b11*k[1] + b12*k[2] +
            b13*k[3]);

    k[5] = y + h*(k[5] + bh10*k[6] + bh11*k[1] + bh12*k[2]);

    // Estimacion del error.
    
    err = 0.e0;
    denom = max(max(1.e-6, fabs(k[4].x)), max(fabs(y.x), 2.e0*uround/epsilon));
    err = err + pown(((k[4].x-k[5].x)/denom), 2);
    
    denom = max(max(1.e-6, fabs(k[4].y)), max(fabs(y.y), 2.e0*uround/epsilon));
    err = err + pown(((k[4].y-k[5].y)/denom), 2);
    
    denom = max(max(1.e-6, fabs(k[4].z)), max(fabs(y.z), 2.e0*uround/epsilon));
    err = err + pown(((k[4].z-k[5].z)/denom), 2);
    
    denom = max(max(1.e-6, fabs(k[4].w)), max(fabs(y.w), 2.e0*uround/epsilon));
    err = err + pown(((k[4].w-k[5].w)/denom), 2);
    err = sqrt(err/4.0);

    //double calculo de hnew
    //.333<=hnew/w<=6.

    fac = max((1.e0/6.e0), min(3.e0, pow((err/epsilon),1.e0/8.e0)/0.9e0));
    hnew = h/fac;

    if(islessequal(err,epsilon)){
   //Paso aceptado 
        naccpt++;
        
        y = k[4];
        *y_b = y;
        t = xph;

        if(fabs(hnew) > hmax){
            hnew = posneg*hmax;
        }
        if(reject){
            hnew = posneg*min(fabs(hnew), fabs(h));
        }
        
        reject = false;
        ok2 = false;
        h = hnew;
    }
    else
    {
        reject = true;
        h = hnew;
        if(naccpt > 1){
            nreject ++;
        }
        nfcn=nfcn-1;
    }

    }

    }
}



__kernel void runge_kutta_8(__global double4 *vector_out, 
        __constant double4 *vector_in, int dimension,
        double initial_time_step, double final_time, __local double4 *local_buffer,
         __global int *exit_c, __global double *time_matrix_out)
{
    int gud_0, lsz_0, lid_0, gsz_0, gid_0;
    gud_0 = get_group_id(0);
    lid_0 = get_local_id(0);
    lsz_0 = get_local_size(0);
    gsz_0 = get_global_size(0);
    gid_0 = get_global_id(0);
    double step_;
    double time_start;
    double4 val;
    time_start = initial_time_step;
    int eco = 0;

    double h= 1.e-2;
    double hmax = 1.e0;
    double error = 1.e-15;
    step_ =(final_time-initial_time_step)/dimension; //step size

    val = vector_in[gid_0];

    for (int i = 0; i <= dimension; i++){
        final_time = fma(i, step_, initial_time_step);
        integration_function(time_start, &val , final_time, error, hmax, h,  exit_c, eco, gid_0); //Exec. integration
        if (gid_0==0){
            /*Only gid_0 = 0 fills time matrix with every step */
            time_matrix_out[i] = final_time;
        }
        /*Output every step of integration.*/
        vector_out[gid_0 + i*gsz_0] = val;
        time_start = final_time;
        
        /*Kill the integration process if error */
        /*
        if (eco == 0){
            ;
        }
        else{
        return ;
            }
        */
    }
   // vector_out[gid_0] = val; //Uncomment only last step. Comment vector_out[gid_0 + i*gsz_0] reduce mem buffer size
}



