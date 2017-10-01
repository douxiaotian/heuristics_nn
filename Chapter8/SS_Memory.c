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

#include "ss.h"

SS *DataStructures_init(int nvar,int b,int PSize,int LocalSearch,int Freq)
{
	SS *problem;

	problem = (SS*) calloc(1,sizeof(SS));
    if(!problem) SSabort("Memory allocation problem");

	problem->n_var  = nvar;
	problem->b1		= b;
	problem->PSize	= PSize;
	problem->LS     = LocalSearch;
	problem->digits = 5; 

	problem->last_combine = 0;
	problem->iter=0;

	problem->ImpCount=0;
	problem->ImpFreq = Freq;

	problem->high  = SSallocate_double_array(nvar);
	if(!problem->high) SSabort("Memory allocation problem");

	problem->low  = SSallocate_double_array(nvar);
	if(!problem->low) SSabort("Memory allocation problem");

	problem->ranges  = SSallocate_int_matrix(nvar,5);
	if(!problem->ranges) SSabort("Memory allocation problem");


	problem->value1  = SSallocate_double_array(b);
	if(!problem->value1) SSabort("Memory allocation problem");

	problem->RefSet1	 = SSallocate_double_matrix(b,nvar);
	if(!problem->RefSet1) SSabort("Memory allocation problem");
	
	problem->order1  = SSallocate_int_array(b);
	if(!problem->order1) SSabort("Memory allocation problem");

	problem->iter1  = SSallocate_int_array(b);
	if(!problem->iter1) SSabort("Memory allocation problem");


	/* Parameters for random generator */
	problem->idum  = 13171191; 
	problem->seed_reset = 1;
	problem->iff		= 0;    

	return problem;
}

void Free_DataStructures(SS *prob)
{
	SSfree_double_matrix(prob->RefSet1,prob->b1);
	free(prob->value1);
	free(prob->order1);
	free(prob->iter1);

	SSfree_int_matrix(prob->ranges,prob->n_var);
	
	free(prob->high);
	free(prob->low);
	free(prob);
}


int **SSallocate_int_matrix(int rows,int columns)
{
	int **aux;
	int i;

	aux=(int**)calloc(rows+1,sizeof(int*));
	if(!aux) SSabort("Memory allocation problem");

	for(i=1;i<=rows;i++)
	{
		aux[i] = SSallocate_int_array(columns);
		if(aux[i]==NULL) SSabort("Memory allocation problem");
	}
	return aux;
}


double **SSallocate_double_matrix(int rows,int columns)
{
	double **aux;
	int i;

	aux=(double**)calloc(rows+1,sizeof(double*));
	if(!aux) SSabort("Memory allocation problem");

	for(i=0;i<=rows;i++)
	{
		aux[i] = SSallocate_double_array(columns);
		if(!aux[i]) SSabort("Memory allocation problem");
	}
	return aux;
}

double *SSallocate_double_array(int size)
{
	double *aux;

	aux=(double*)calloc(size+1,sizeof(double));
	if(!aux) SSabort("Memory allocation problem");
	return aux;
}


int *SSallocate_int_array(int size)
{
	int *aux;

	aux=(int*)calloc(size+1,sizeof(int));
	if(!aux) SSabort("Memory allocation problem");
	return aux;
}


void SSfree_double_matrix(double **matrix,int rows)
{
	int i;

	for(i=0;i<=rows;i++)
		free(matrix[i]);
	free(matrix);
}

void SSfree_int_matrix(int **matrix,int rows)
{
	int i;

	for(i=1;i<=rows;i++)
		free(matrix[i]);
	free(matrix);
}




