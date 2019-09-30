//============================================================//
//                                                            //
//   This codes provides parallel omp codes to compute the    //
//  optimal fit for given molecular spectra, using the NLOPT  //
//     package. In addition it computes the optimal Raman     //
//  assisted excitation field to maximize discrimination in   //
//   a given ensemble of biological molecules for which the   //
//      excitation and Raman spectra are known or can be      //
//                  approximately obtained.                   //
//                                                            //
//                @author  A. Chattopadhyay                   //
//    @affiliation Princeton University, Dept. of Chemistry   //
//           @version Updated last on Dec 14 2018             //
//                                                            //
//============================================================//

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nlopt.h>
#include <omp.h>
#include <time.h>
#define ENERGY_FACTOR 1. / 27.211385
#define ERROR_BOUND 1.0E-8
#define WAVELENGTH2FREQ 1239.84
#define NLOPT_XTOL 1.0E-6

typedef double complex cmplx;

typedef struct parameters{

    cmplx* rho_0;
    int nDIM;
    int N_exc;
    double* time;
    double timeAMP_A;
    double timeAMP_R;
    double timeAMP;
    int timeDIM;
    double delay;
    double field_amp_A;
    double field_amp_R;
    double omega_R;
    double omega_v;
    double omega_e;
    double d_alpha;
    int thread_num;
    int prob_guess_num;
    double* spectra_lower;
    double* spectra_upper;
    int max_iter;
    double* control_guess;
    double* control_lower;
    double* control_upper;
    int guess_num;
    int max_iter_control;
} parameters;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* matrix_gamma_pd;
    double* matrix_gamma_dep;
    double gamma_dep;
    double* frequency_A;
    int freqDIM_A;
    cmplx* rho_0;
    cmplx* mu;
    cmplx* field_R;
    cmplx* field_A;
    cmplx* field;
    cmplx* rho;
    double* abs_spectra;
    double* abs_dist;
    double* ref_spectra;
    double* Raman_levels;
    double* levels;
    cmplx* dyn_rho;
    double* prob;
} molecule;

typedef struct mol_system{
    molecule** ensemble;
    molecule* original;
    parameters* params;
    int* count;
} mol_system;

typedef struct supersystem{
    mol_system* mol_system_A;
    mol_system* mol_system_B;
    int* count;
} supersystem;


//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a matrix or vector and their dimensionality, these routines perform the operations of printing, adding,   //
//        scaling, copiesing to another compatible data structure, finding trace, or computing the maximum element.   //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(const cmplx *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX VECTOR                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(const double *A, const int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(const double *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A REAL VECTOR                    //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_mat(const cmplx *A, cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM1; i++)
    {
        for(j=0; j<nDIM2; j++)
        {
            B[i * nDIM2 + j] += A[i * nDIM2 + j];
        }
    }
}


void add_vec(const double *A, double *B, const int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        B[i] += A[i];
    }
}

void scale_mat(cmplx *A, const double factor, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM1; i++)
    {
        for(int j=0; j<nDIM2; j++)
        {
            A[i * nDIM2 + j] *= factor;
        }
    }
}


void scale_vec(double *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> VECTOR B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        A[i] *= factor;
    }
}


cmplx complex_trace(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


double complex_max_element(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}

double vec_max(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    for(int i=0; i<nDIM; i++)
    {
        if(A[i] > max_el)
        {
            max_el = A[i];
        }
    }
    return max_el;
}

double vec_sum(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	            RETURNS SUM OF VECTOR ELEMENTS        //
//----------------------------------------------------//
{
    double sum = 0.0;
    for(int i=0; i<nDIM; i++)
    {
        sum += A[i];
    }
    return sum;
}


double vec_diff_norm(const double *const A, const double *const B, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS L-1 NORM OF VECTOR DIFFERENCE          //
//----------------------------------------------------//
{
    int nfrac = (int)(nDIM/2.2);
    double norm = 0.0;
    double norm_long_wavelength = 0.0;
    for(int i=0; i<nfrac; i++)
    {
        norm_long_wavelength += fabs(A[i]-B[i]);
    }

    for(int i=0; i<nDIM; i++)
    {
        norm += fabs(A[i]-B[i]);
    }
    printf("norm = %g, norm1 = %g \n", norm, norm_long_wavelength);
//    return norm*norm_long_wavelength;
    return norm;
}



//====================================================================================================================//
//                                                                                                                    //
//                   CALCULATION OF FIELDS FOR ABSORPTION AND RAMAN SPECTRA CALCULATIONS                              //
//                                                                                                                    //
//====================================================================================================================//

void CalculateAbsorptionSpectraField(molecule* mol, const parameters *const params, const int k)
//--------------------------------------------------------------------------------//
//     RETURNS THE ABSORPTION SPECTRA CALCULATION FIELD AS A FUNCTION OF TIME     //
//             k ----> index corresponding to spectral wavelength                 //
//--------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    double timeAMP_A = params->timeAMP_A;

    double* t = params->time;
    const double dt = fabs(t[1] - t[0]);
    double A = params->field_amp_A;

    for(i=0; i<params->timeDIM; i++)
    {
        const double time = t[i] + 0.5 * dt;
        mol->field[i] = A * pow(cos(M_PI*time/timeAMP_A), 2) * cos(mol->frequency_A[k] * time);
    }
}

void CalculateControlField(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------//
//                 RETURNS THE RAMAN CONTROL FIELD AS A FUNCTION OF TIME           //
//---------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM;

    double timeAMP_A = params->timeAMP_A;
    double timeAMP_R = params->timeAMP_R;
    double timeAMP = params->timeAMP;

    double delay = params->delay;

    int delayINDX = (int) (timeDIM * (timeAMP_R - delay) / timeAMP);
    int timeDIM_R = (int) (timeDIM * timeAMP_R / timeAMP);
    int timeDIM_A = (int) (timeDIM * (timeAMP_A) / timeAMP);

    double* t = params->time;
    const double dt = params->timeAMP / (params->timeDIM - 1);

    double A_R = params->field_amp_R;
    double A_E = params->field_amp_A;

    for(i=0; i<timeDIM_R; i++)
    {
        const double time = t[i] + 0.5 * dt;
        mol->field_R[i] = A_R * pow(cos(M_PI*(time - timeAMP_R / 2)/(timeAMP_R)), 2) *
                          ((cos((params->omega_R + params->omega_v) * time)) + cos(params->omega_R * time));
    }

    for(i=delayINDX; i<delayINDX + timeDIM_A; i++)
    {
        const double time = t[i] + 0.5 * dt;
        mol->field_A[i] = A_E * pow(cos(M_PI*(time - timeAMP_R - timeAMP_A / 2 + delay)/timeAMP_A), 2) * cos(params->omega_e * time);
    }

    for(i=0; i<timeDIM; i++)
    {
        mol->field[i] = mol->field_R[i] + mol->field_A[i];
    }
}


//====================================================================================================================//
//                                                                                                                    //
//                                CALCULATION OF OPEN QUANTUM SYSTEM DYNAMICS                                         //
//                                                                                                                    //
//====================================================================================================================//


void L_operate(cmplx* Qmat, const cmplx field_ti, molecule* mol, const parameters *const params)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    const int nDIM = mol->nDIM;
    double* matrix_gamma_dep = mol->matrix_gamma_dep;
    double* matrix_gamma_pd = mol->matrix_gamma_pd;
    cmplx* mu = mol->mu;
    double* energies = mol->energies;

    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));

    for(int m = 0; m < nDIM; m++)
        {
        for(int n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] -= I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(int k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + n] += I * field_ti * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);
                    Lmat[m * nDIM + n] += I * params->d_alpha * field_ti * field_ti * (Qmat[k * nDIM + n] - Qmat[m * nDIM + k]);

                    Lmat[m * nDIM + n] -= 0.5 * (matrix_gamma_pd[k * nDIM + n] + matrix_gamma_pd[k * nDIM + m]) * Qmat[m * nDIM + n];
                    Lmat[m * nDIM + n] += matrix_gamma_pd[m * nDIM + k] * Qmat[k * nDIM + k];
                }

                Lmat[m * nDIM + n] -= matrix_gamma_dep[m * nDIM + n] * Qmat[m * nDIM + n];
            }

        }

    for(int m = 0; m < nDIM; m++)
        {
        for(int n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }
    free(Lmat);

}

//====================================================================================================================//
//                                                                                                                    //
//                                  PROPAGATION STEP FOR A GIVEN WAVELENGTH                                           //
//                                                                                                                    //
//====================================================================================================================//


void PropagateAbsorptionSpectra(molecule* mol, const parameters *const params, const int index)
//--------------------------------------------------------------------------------------------------------------------//
// 	 	 		       CALCULATES FULL LINDBLAD DYNAMICS  DUE TO THE CONTROL FIELD FROM TIME 0 to T               	  //
//                            indx gives index of the specific wavelength in the spectra                              //
//--------------------------------------------------------------------------------------------------------------------//
{

    int t_i, k;
    const int nDIM = params->nDIM;

    cmplx *rho_0 = mol->rho_0;
    double *time = params->time;
    double dt = params->timeAMP / (params->timeDIM - 1);

    cmplx* field = mol->field;

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_i=0; t_i<params->timeDIM; t_i++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_i], mol, params);
            scale_mat(L_rho_func, dt/k, nDIM, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > 1.0E-8);

        copy_mat(mol->rho, L_rho_func, nDIM);
    }

    for(int j=1; j<=params->N_exc; j++)
    {
        mol->abs_spectra[index] += mol->rho[(nDIM-j)*nDIM + (nDIM-j)];
    }
    free(L_rho_func);
}

void Control(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------------------------------------------//
// 	 	 		     CALCULATES FULL LINDBLAD DYNAMICS  DUE TO THE RAMAN CONTROL FIELD FROM TIME 0 to T               //
//--------------------------------------------------------------------------------------------------------------------//
{
    int i, k, t_index;
    const int nDIM = mol->nDIM;

    cmplx *rho_0 = mol->rho_0;
    double *time = params->time;

    cmplx* field = mol->field;

    double dt = params->timeAMP / (params->timeDIM - 1);

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_index=0; t_index<params->timeDIM; t_index++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_index], mol, params);
            scale_mat(L_rho_func, dt/k, nDIM, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > ERROR_BOUND);

        for(i=0; i<nDIM; i++)
        {
            mol->dyn_rho[i*params->timeDIM + t_index] = mol->rho[i*nDIM + i];
        }

        copy_mat(mol->rho, L_rho_func, nDIM);
    }

    free(L_rho_func);
}


void copy_molecule(molecule* original, molecule* copy, parameters* params)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    int N = params->prob_guess_num;
    int nDIM = params->nDIM;

    copy->energies = (double*)malloc(N*sizeof(double));
    copy->nDIM = original->nDIM;
    copy->matrix_gamma_pd = (double*)malloc(nDIM*nDIM*sizeof(double));
    copy->matrix_gamma_dep = (double*)malloc(nDIM*nDIM*sizeof(double));
    copy->gamma_dep = original->gamma_dep;
    copy->frequency_A = (double*)malloc(original->freqDIM_A*sizeof(double));
    copy->freqDIM_A = original->freqDIM_A;
    copy->rho_0 = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
    copy->mu = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
    copy->field_A = (cmplx*)malloc(params->timeDIM*sizeof(cmplx));
    copy->field_R = (cmplx*)malloc(params->timeDIM*sizeof(cmplx));
    copy->field = (cmplx*)malloc(params->timeDIM*sizeof(cmplx));
    copy->rho = (cmplx*)malloc(nDIM*nDIM*sizeof(cmplx));
    copy->abs_spectra = (double*)malloc(original->freqDIM_A*sizeof(double));
    copy->abs_dist = (double*)malloc(params->prob_guess_num*original->freqDIM_A*sizeof(double));
    copy->ref_spectra = (double*)malloc(original->freqDIM_A*sizeof(double));
    copy->Raman_levels = (double*)malloc((nDIM - params->N_exc)*sizeof(double));
    copy->levels = (double*)malloc(params->N_exc*params->prob_guess_num*sizeof(double));
    copy->dyn_rho = (cmplx*)malloc(nDIM*params->timeDIM*sizeof(cmplx));
    copy->prob = (double*)malloc(N*sizeof(double));

    memset(copy->energies, 0, params->nDIM*sizeof(double));
    memcpy(copy->matrix_gamma_pd, original->matrix_gamma_pd, nDIM*nDIM*sizeof(double));
    memcpy(copy->matrix_gamma_dep, original->matrix_gamma_dep, nDIM*nDIM*sizeof(double));
    memcpy(copy->frequency_A, original->frequency_A, original->freqDIM_A*sizeof(double));
    memcpy(copy->rho_0, original->rho_0, nDIM*nDIM*sizeof(cmplx));
    memcpy(copy->mu, original->mu, nDIM*nDIM*sizeof(cmplx));
    memcpy(copy->field_A, original->field_A, params->timeDIM*sizeof(cmplx));
    memcpy(copy->field_R, original->field_R, params->timeDIM*sizeof(cmplx));
    memcpy(copy->field, original->field, params->timeDIM*sizeof(cmplx));
    memcpy(copy->rho, original->rho, nDIM*nDIM*sizeof(cmplx));
    memcpy(copy->abs_spectra, original->abs_spectra, original->freqDIM_A*sizeof(double));
    memcpy(copy->abs_dist, original->abs_dist, params->prob_guess_num*original->freqDIM_A*sizeof(double));
    memcpy(copy->ref_spectra, original->ref_spectra, original->freqDIM_A*sizeof(double));
    memcpy(copy->Raman_levels, original->Raman_levels, (nDIM - params->N_exc)*sizeof(double));
    memcpy(copy->levels, original->levels, params->N_exc*params->prob_guess_num*sizeof(double));
    memcpy(copy->dyn_rho, original->dyn_rho, nDIM*params->timeDIM*sizeof(cmplx));
    memcpy(copy->prob, original->prob, N*sizeof(double));
}


double nloptJ_spectra(unsigned N, const double *opt_spectra_params, double *grad_J, void *nloptJ_spectra_params)
{
    mol_system* system = (mol_system*)nloptJ_spectra_params;

    parameters* params = system->params;
    molecule** ensemble = system->ensemble;
    molecule* mol = system->original;
    int* count = system->count;

    int nDIM = params->nDIM;

    memset(mol->abs_spectra, 0, mol->freqDIM_A*sizeof(double));

    #pragma omp parallel for
    for(int j=0; j<params->prob_guess_num; j++)
    {
        memset(ensemble[j]->abs_spectra, 0, mol->freqDIM_A*sizeof(double));
        for(int i=0; i<mol->freqDIM_A; i++)
        {
            CalculateAbsorptionSpectraField(ensemble[j], params, i);
            PropagateAbsorptionSpectra(ensemble[j], params, i);
        }
        scale_vec(ensemble[j]->abs_spectra, opt_spectra_params[j], mol->freqDIM_A);
        add_vec(ensemble[j]->abs_spectra, mol->abs_spectra, mol->freqDIM_A);

    }

    for(int j=0; j<params->prob_guess_num; j++)
    {
        for(int k=0; k<mol->freqDIM_A; k++)
        {
            mol->abs_dist[j*mol->freqDIM_A + k] = 100. * ensemble[j]->abs_spectra[k] / vec_max(mol->abs_spectra, mol->freqDIM_A);
        }

    }
    scale_vec(mol->abs_spectra, 100./vec_max(mol->abs_spectra, mol->freqDIM_A), mol->freqDIM_A);
    double J;
    J = vec_diff_norm(mol->ref_spectra, mol->abs_spectra, mol->freqDIM_A);

    *count = *count + 1;
    printf("%d | (", *count);
    for(int i=0; i<params->prob_guess_num; i++)
    {
        printf("%3.2lf ", opt_spectra_params[i]);
    }
    printf(")  fit = %3.5lf \n", J);
    return J;
}


cmplx* CalculateSpectra(molecule* mol, parameters* params)
//------------------------------------------------------------//
//          CALCULATING SPECTRAL FIT FOR A MOLECULE           //
//------------------------------------------------------------//
{

    // ---------------------------------------------------------------------- //
    //      UPDATING THE PURE DEPHASING MATRIX & ENERGIES FOR MOLECULE        //
    // ---------------------------------------------------------------------- //

    int N_vib = mol->nDIM - params->N_exc;

    molecule** ensemble = (molecule**)malloc(params->prob_guess_num * sizeof(molecule*));
    for(int i=0; i<params->prob_guess_num; i++)
    {
        ensemble[i] = (molecule*)malloc(sizeof(molecule));
        copy_molecule(mol, ensemble[i], params);
        for(int j=0; j<N_vib; j++)
        {
            ensemble[i]->energies[j] = mol->Raman_levels[j];
            for(int k=N_vib; k<mol->nDIM; k++)
            {
                ensemble[i]->matrix_gamma_dep[j*mol->nDIM + k] = mol->gamma_dep;
                ensemble[i]->matrix_gamma_dep[k*mol->nDIM + j] = mol->gamma_dep;
            }
        }

        for(int j=0; j<params->N_exc; j++)
        {
            ensemble[i]->energies[N_vib + j] = mol->levels[params->N_exc*i+j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //

    mol_system system;
    system.ensemble = ensemble;
    system.original = (molecule*)malloc(sizeof(molecule));
    memcpy(system.original, mol, sizeof(molecule));
    system.params = params;
    system.count = (int*)malloc(sizeof(int));
    *system.count = 0;

    // ---------------------------------------------------------------------- //
    //              INITIALIZING NLOPT CLASS AND PARAMETERS                   //
    // ---------------------------------------------------------------------- //

    nlopt_opt opt_spectra;

    double *spectra_lower = params->spectra_lower;
    double *spectra_upper = params->spectra_upper;

    opt_spectra = nlopt_create(NLOPT_LN_COBYLA, params->prob_guess_num);  // Local no-derivative optimization algorithm
//    opt_spectra = nlopt_create(NLOPT_GN_DIRECT, params->prob_guess_num);  // Global no-derivative optimization algorithm
    nlopt_set_lower_bounds(opt_spectra, spectra_lower);
    nlopt_set_upper_bounds(opt_spectra, spectra_upper);
    nlopt_set_min_objective(opt_spectra, nloptJ_spectra, (void*)&system);
    nlopt_set_xtol_rel(opt_spectra, NLOPT_XTOL);
    nlopt_set_maxeval(opt_spectra, params->max_iter);

    double *x = mol->prob;
    double minf;

    if (nlopt_optimize(opt_spectra, x, &minf) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at ( ");

        for(int m=0; m<params->prob_guess_num; m++)
        {
            printf(" %g,", x[m]);
        }
        printf(") = %g\n", minf);
    }

    nlopt_destroy(opt_spectra);

}

double nloptJ_control(unsigned N, const double *opt_control_params, double *grad_J, void *nloptJ_control_params)
{
    supersystem* molecules = (supersystem*)nloptJ_control_params;

    parameters* params = molecules->mol_system_A->params;
    molecule** ensemble_A = molecules->mol_system_A->ensemble;
    molecule** ensemble_B = molecules->mol_system_B->ensemble;
    molecule* mol_A = molecules->mol_system_A->original;
    molecule* mol_B = molecules->mol_system_B->original;
    int* count = molecules->count;

    double dt;

    double* prob_A = (double*)malloc(params->prob_guess_num*sizeof(double));
    double* prob_B = (double*)malloc(params->prob_guess_num*sizeof(double));
    memcpy(prob_A, mol_A->prob, params->prob_guess_num*sizeof(double));
    memcpy(prob_B, mol_B->prob, params->prob_guess_num*sizeof(double));

    scale_vec(prob_A, 1.0 / vec_sum(prob_A, params->prob_guess_num), params->prob_guess_num);
    scale_vec(prob_B, 1.0 / vec_sum(prob_B, params->prob_guess_num), params->prob_guess_num);

    params->field_amp_R = opt_control_params[0];
    params->field_amp_A = opt_control_params[1];
    params->omega_R = opt_control_params[2];
    params->omega_v = opt_control_params[3];
    params->omega_e = opt_control_params[4];
    params->d_alpha = opt_control_params[5];
    params->timeAMP_A = opt_control_params[6];
    params->timeAMP_R = opt_control_params[7];

    dt = params->timeAMP/ (params->timeDIM - 1);

    for(int i=0; i<params->timeDIM; i++)
    {
        params->time[i] = i * dt;
    }

    memset(mol_A->dyn_rho, 0, mol_A->nDIM*params->timeDIM*sizeof(cmplx));
    memset(mol_B->dyn_rho, 0, mol_B->nDIM*params->timeDIM*sizeof(cmplx));
    memset(mol_A->rho, 0, mol_A->nDIM*mol_A->nDIM*sizeof(cmplx));
    memset(mol_B->rho, 0, mol_B->nDIM*mol_B->nDIM*sizeof(cmplx));

    CalculateControlField(mol_A, params);
    CalculateControlField(mol_B, params);

    #pragma omp parallel for
    for(int j=0; j<params->prob_guess_num; j++)
    {
        memset(ensemble_A[j]->dyn_rho, 0, mol_A->nDIM*params->timeDIM*sizeof(cmplx));
        memset(ensemble_B[j]->dyn_rho, 0, mol_B->nDIM*params->timeDIM*sizeof(cmplx));

        memcpy(ensemble_A[j]->field_R, mol_A->field_R, params->timeDIM*sizeof(cmplx));
        memcpy(ensemble_B[j]->field_R, mol_B->field_R, params->timeDIM*sizeof(cmplx));
        memcpy(ensemble_A[j]->field_A, mol_A->field_A, params->timeDIM*sizeof(cmplx));
        memcpy(ensemble_B[j]->field_A, mol_B->field_A, params->timeDIM*sizeof(cmplx));
        memcpy(ensemble_A[j]->field, mol_A->field, params->timeDIM*sizeof(cmplx));
        memcpy(ensemble_B[j]->field, mol_B->field, params->timeDIM*sizeof(cmplx));

        Control(ensemble_A[j], params);
        Control(ensemble_B[j], params);

        scale_mat(ensemble_A[j]->dyn_rho, prob_A[j], mol_A->nDIM, params->timeDIM);
        add_mat(ensemble_A[j]->dyn_rho, mol_A->dyn_rho, mol_A->nDIM, params->timeDIM);
        scale_mat(ensemble_B[j]->dyn_rho, prob_B[j], mol_B->nDIM, params->timeDIM);
        add_mat(ensemble_B[j]->dyn_rho, mol_B->dyn_rho, mol_B->nDIM, params->timeDIM);

        scale_mat(ensemble_A[j]->rho, prob_A[j], mol_A->nDIM, mol_A->nDIM);
        add_mat(ensemble_A[j]->rho, mol_A->rho, mol_A->nDIM, mol_A->nDIM);
        scale_mat(ensemble_B[j]->rho, prob_B[j], mol_B->nDIM, mol_B->nDIM);
        add_mat(ensemble_B[j]->rho, mol_B->rho, mol_B->nDIM, mol_B->nDIM);
    }

    int nDIM = mol_A->nDIM;
    int N_vib = nDIM - params->N_exc;
    double pop_exc_A = 0.0;
    double pop_exc_B = 0.0;
    double J;

    for(int i=N_vib; i<nDIM; i++)
    {
        pop_exc_A += creal(mol_A->rho[i*nDIM + i]);
        pop_exc_B += creal(mol_B->rho[i*nDIM + i]);
    }

    J = pop_exc_A / pop_exc_B;

    *count = *count + 1;
//    printf("%d| (", *count);
//    for(int i=0; i<params->guess_num; i++)
//    {
//        printf("%3.6lf ", opt_control_params[i]);
//    }
//    printf(")  fit = %3.6lf ; %3.6lf ; %g ; %g\n", J, pop_exc_A / pop_exc_B, pop_exc_A, pop_exc_B);
    return J;
}

cmplx* CalculateControl(molecule* mol_A, molecule* mol_B, parameters* params)
//------------------------------------------------------------//
//              CALCULATING RAMAN CONTROL PARAMETERS          //
//------------------------------------------------------------//
{
    // ---------------------------------------------------------------------- //
    //              INITIALIZING ORIGINAL MOLECULE AND ENSEMBLES              //
    // ---------------------------------------------------------------------- //

    int N_vib = mol_A->nDIM - params->N_exc;
    molecule** ensemble_A = (molecule**)malloc(params->prob_guess_num * sizeof(molecule*));
    molecule** ensemble_B = (molecule**)malloc(params->prob_guess_num * sizeof(molecule*));
    for(int i=0; i<params->prob_guess_num; i++)
    {
        ensemble_A[i] = (molecule*)malloc(sizeof(molecule));
        ensemble_B[i] = (molecule*)malloc(sizeof(molecule));
        copy_molecule(mol_A, ensemble_A[i], params);
        copy_molecule(mol_B, ensemble_B[i], params);
        for(int j=0; j<N_vib; j++)
        {
            ensemble_A[i]->energies[j] = mol_A->Raman_levels[j];
            ensemble_B[i]->energies[j] = mol_B->Raman_levels[j];
            for(int k=N_vib; k<mol_A->nDIM; k++)
            {
                ensemble_A[i]->matrix_gamma_dep[j*mol_A->nDIM + k] = mol_A->gamma_dep;
                ensemble_B[i]->matrix_gamma_dep[j*mol_A->nDIM + k] = mol_B->gamma_dep;
                ensemble_A[i]->matrix_gamma_dep[k*mol_A->nDIM + j] = mol_A->gamma_dep;
                ensemble_B[i]->matrix_gamma_dep[k*mol_A->nDIM + j] = mol_B->gamma_dep;
            }
        }

        for(int j=0; j<params->N_exc; j++)
        {
            ensemble_A[i]->energies[N_vib + j] = mol_A->levels[params->N_exc*i+j];
            ensemble_B[i]->energies[N_vib + j] = mol_B->levels[params->N_exc*i+j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLES OF MOLECULES                  //
    // ---------------------------------------------------------------------- //

    supersystem molecules;

    mol_system system_A;
    system_A.ensemble = ensemble_A;
    system_A.original = (molecule*)malloc(sizeof(molecule));
    memcpy(system_A.original, mol_A, sizeof(molecule));
    system_A.params = params;
    system_A.count = (int*)malloc(sizeof(int));
    *system_A.count = 0;

    mol_system system_B;
    system_B.ensemble = ensemble_B;
    system_B.original = (molecule*)malloc(sizeof(molecule));
    memcpy(system_B.original, mol_B, sizeof(molecule));
    system_B.params = params;
    system_B.count = (int*)malloc(sizeof(int));
    *system_B.count = 0;

    molecules.mol_system_A = &system_A;
    molecules.mol_system_B = &system_B;
    *molecules.count = 0;

    // ---------------------------------------------------------------------- //
    //              INITIALIZING NLOPT CLASS AND PARAMETERS                   //
    // ---------------------------------------------------------------------- //

    nlopt_opt opt_control;

    double *control_lower = params->control_lower;
    double *control_upper = params->control_upper;

    opt_control = nlopt_create(NLOPT_LN_COBYLA, params->guess_num);  // Local no-derivative optimization algorithm
//    opt_control = nlopt_create(NLOPT_GN_DIRECT, params->guess_num);  // Global no-derivative optimization algorithm
    nlopt_set_lower_bounds(opt_control, control_lower);
    nlopt_set_upper_bounds(opt_control, control_upper);
    nlopt_set_max_objective(opt_control, nloptJ_control, (void*)&molecules);
    nlopt_set_xtol_rel(opt_control, NLOPT_XTOL);
    nlopt_set_maxeval(opt_control, params->max_iter_control);

    double *x = params->control_guess;
    double maxf;

    if (nlopt_optimize(opt_control, x, &maxf) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("maximum at ( ");

        for(int m=0; m<params->guess_num; m++)
        {
            printf(" %g,", x[m]);
        }
        printf(") = %g\n", maxf);
    }

    nlopt_set_maxeval(opt_control, 1);
    nlopt_optimize(opt_control, x, &maxf);
    nlopt_destroy(opt_control);
}