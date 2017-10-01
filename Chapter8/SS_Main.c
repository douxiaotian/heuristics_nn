/********************************************/
/*                                          */
/*  Scatter Search C code					*/
/*	Neural Network Prediction in a System	*/
/*  for Optimizing Simulations              */
/*                                          */
/*  Authors: M. Laguna and R. Martí         */
/*  Copyright © 2000                        */
/*                                          */
/********************************************/

#include <time.h>
#include "ss.h"

#define  MAX(X,Y)   ( (X) > (Y) ? (X) : (Y) )
#define  MIN(X,Y)   ( (X) < (Y) ? (X) : (Y) )
long EvalNum = 0;        /* current number of evaluations      */

int main(int argc, char **argv)

{
	int    i;
	int    nvar;         /* number of input variables          */
	int    np;           /* function (problem) number          */
	int    b;            /* size of the reference set          */
	int    PSize;        /* size of diversification set        */
	int    LocalSearch;  /* local search switch (1=ON, 0=OFF)  */
	int    m;            /* neurons in the hidden layer        */
	int    train_size;   /* size of the training set           */
	int    regression;   /* regression switch (1=ON, 0=OFF)    */
	int    scaling;      /* scaling switch  (1=ON, 0=OFF)      */
	int    activation;   /* activation function switch        
	                        (1=sigmoid, 2= tanh, 3=identity    */
	int    ImpFreq;      /* local search use frequency         */
	int    TotalEval;    /* total number of evaluations (50000)*/
	double *train_value; /* output values in the training set  */
	double **train_data; /* input values in the training set   */
	double wlow;         /* lower bound for diverse w          */
	double whigh;        /* upper bound for diverse w          */
	double error;        /* MSE                                */
	double pred;         /* prediction value                   */
	SS     *prob;        /* scatter search structure           */
	Net     *p;          /* NN structure                       */
	clock_t start;

	if (argc != 2) {
		printf("usage: program_name problem_number (1 to 6\n");
		exit(1);
	}

	/* Neural Net Parameters */
	np			= atoi(argv[1]);
	m			= 6;
	train_size  = 50;
	scaling     = 1;
	activation  = 1;
	regression  = 1;

	/* Scatter Search Parameters */
	TotalEval  = 30000;
	wlow       =  -2; 
	whigh      =   2; 
	b		   =  10;
 	PSize	   =  100; 
	LocalSearch=   1;
	ImpFreq	   = 200; 

	/* Initializations */
	start = clock();
	srand(11);
	train_value=SSallocate_double_array(train_size);
	train_data=Input_data(np,train_size,&nvar,train_value);
	p=InitNet(nvar,m,train_size,train_data,train_value,
		      regression,scaling,activation);
	prob=DataStructures_init(p->dim,b,PSize,LocalSearch,ImpFreq);

	// Assign Variable Bounds 
	for(i=1;i<=p->dim;i++) { prob->low[i]=wlow; prob->high[i]=whigh; }

	// Build Reference Set 
	Initiate_RefSet(p,prob);

	// Perform Search Procedure
	i = 0;
	while (EvalNum < TotalEval && prob->value1[prob->order1[1]] > EPSILON)
	{
		++i;
		Combine_RefSet(p,prob);
		Intensify(p,4,prob->RefSet1[prob->order1[1]],&(prob->value1[prob->order1[1]]));
		Update_RefSet(p,prob);
	}

	for(i=1;i<=p->dim;i++) 
		p->w[i]=prob->RefSet1[prob->order1[1]][i];

	// Compute Interpolation Error
	SSfree_double_matrix(train_data,train_size);
	free(train_value);
	train_size=150;
	train_value=SSallocate_double_array(train_size);
	train_data=Input_data(np,train_size,&nvar,train_value);
	error=0;
	for(i=1;i<=train_size;i++) {
		pred = net_prediction(p,train_data[i]);
		error += pow(pred-train_value[i],2);
	}
	error /= train_size;

	// Print results
	printf("Problem #%d \nEvaluations=%ld",np,EvalNum);
	printf("\nCPU=%5.1f seconds",(clock()-start)/(float)CLOCKS_PER_SEC);
	printf("\nTraining Error=%4.3E", sqrt(prob->value1[prob->order1[1]]));
	printf("\nInterpolation Error=%4.3E ",sqrt(error));
	
	Free_DataStructures(prob);
}


