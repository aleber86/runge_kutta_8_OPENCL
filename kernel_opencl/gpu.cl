#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64
#include"coeficient_block.h"

double4 diff_eq(double4 arg, double time_value){
    double4 result;
    result.x = 2.0*time_value;
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

    double h= 1.e-3;
    double hmax = 1.e0;
    double error = 1.e-15;
    step_ =(final_time-initial_time_step)/dimension; //step size

    val = vector_in[gid_0];

    for (int i = 0; i <= dimension; i++){
        final_time = fma(i, step_, initial_time_step);
        integration_function(time_start, &val , final_time, error, hmax, h,  exit_c, eco, gid_0); //Exec. integration
        time_start = final_time;
        /*Kill the integration process if error */
        if (eco == 0){
            ;
        }
        else{
        return ;
            }
    }
   vector_out[gid_0] = val; //Uncomment only last step. Comment vector_out[gid_0 + i*gsz_0] reduce mem buffer size
}



