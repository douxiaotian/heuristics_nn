#include "stdio.h"
#include "stdlib.h"
#include "malloc.h"
#include "math.h"
#include "string.h"
#include "time.h"

typedef struct SS {
	int n_var;	
	double digits;
	double *high;
	double *low;
	int **ranges; /* Diversification Generator */
	int PSize;
	int LS;		  /* =1 LocalSearch ON, 0 OFF */
	int iter;

	int b1;
	double **RefSet1;// Solutions 
	double *value1;  // Objective value
	int *order1;	 //	Order of solutions
	int *iter1;		 // Number of iter of each sol.

	int last_combine;  //Number of iter of last solution combination
	int new_elements;  //True if new elem. added since last combine

	/* Random number parameters */
	long idum;	    				
	int seed_reset;	
	int iff;									
	long ir[98];
	long iy;

	int ImpCount;
	int ImpFreq;

} SS;


typedef struct Net {
	int n1;
	int n2;
	int dim;
	double *w;
	int train_size;
	double **train_set;
	double *train_val;
	double *min_var;
	double *max_var;
	double *offset;
	double *scalei;
	double min_val;
	double max_val;
	double offset_val;
	double scale_val;
	int regression;  // 0 OFF, 1 ON
	int scale;       // 0 OFF, 1 ON
	int activa;		 // 1 for sigmoid, 2 for tanh, 3 for identity 
	int nmax;		 // Maximum number for nmsimplex
} Net;

#define getrandom(min,max)		((rand() % (int)(((max)+1) - (min))) + (min))
#define DBL_MAX 	1.7976931348623158e+308 /* max value */
#define EPSILON     1E-06

static double dmaxarg1,dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1)>(dmaxarg2)?(dmaxarg1) : (dmaxarg2))

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

/* File SS_RefSet.c */
void Intensify(Net *p,int num_iter,double *center,double *besterror);
void Initiate_RefSet(Net *p,SS *prob);
double SSGenerate_value(SS *prob,int a);
void SSimprove_solution(Net *net,double *x,double *value);
void Update_RefSet(Net *p,SS *prob);
void Combine_RefSet(Net *p,SS *prob);

/* File SS_Memory.c */

SS *DataStructures_init(int nvar,int b,int PSize,int LocalSearch,int Freq);
void Free_DataStructures(SS *prob);
int **SSallocate_int_matrix(int rows,int columns);
double **SSallocate_double_matrix(int rows,int columns);
double *SSallocate_double_array(int size);
int *SSallocate_int_array(int size);
void SSfree_double_matrix(double **matrix,int rows);
void SSfree_int_matrix(int **matrix,int rows);


/* File SS_tools.c */

int *orden_indices(double *pesos,int num,int tipo);
void SSabort(char *texto);
double distance_to_RefSet(SS *prob,double *sol);
double distance_to_RefSet1(SS *prob,double *sol);
int is_new(SS *prob,double **solutions,int dim,double *sol);
void SScombine(SS *prob,double *x,double *y,double **offsprings,int number);
void try_add_RefSet1(Net *p,SS *prob,double *sol,double value);


/* File SS_Improve.c */
void SS_Simplex(double **simplex,double *values,int nvar,int max_eval,Net *net);
double SSMove(int nvar,double *worst_point,double *worst_value, double *cum_simplex, double factor,Net *net);

/* File Net */
Net *InitNet(int n1,int n2,int train_size,double **train_data,double *train_value,
			 int reg,int scale,int activation);
double compute_error(Net *p,double *w);
double activation(Net *p,double x);
double net_prediction(Net *p,double *input);

/* Data.c */
double **Input_data(int np, int train_size,int *nvar,double *train_value);
double funcion(int np, double *x);

/* qr.c */
void qrdcmp(double **a,int n,double *c,double *d,int *sing);
void qrsolv(double **a,int n,double c[],double d[],double b[]);
void rsolv(double **a,int n,double d[],double b[]);


