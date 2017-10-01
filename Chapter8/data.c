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
	
double **Input_data(int np, int train_size,int *nvar,double *train_value)
{
	int    i, j, n;
	double **training_set;
	double *y;
	FILE   *fp;

	switch (np)
	{
		case 1:
		case 2:
		case 3:
		case 4:
		case 5: *nvar = n = 2;
				training_set = SSallocate_double_matrix(train_size,n);
				for(i=1;i<=train_size;i++) {
					training_set[i][1] = getrandom(-100,100);
					training_set[i][2] = getrandom(-10,10);
					train_value[i] = funcion(np, training_set[i]);
				}
				break;
		case 6: *nvar = n = 5;
				y = SSallocate_double_array(n*train_size);
				training_set = SSallocate_double_matrix(train_size,n);
				y[1] = 1.6; y[2] = y[3] = y[4] = y[5] = 0.0;
				for(i=6;i<=n*train_size;i++)
					y[i] = y[i-1]+10.5*((0.2*y[i-5])/(1+pow(y[i-5],10))-0.1*y[i-1]);
				for(i=1;i<=train_size;i++) {
					for(j=1;j<=n;++j) 
						training_set[i][j] = y[(i-1)*n+j];
					train_value[i] = y[i*n+1];
				}
				free(y);
				break;

		case 7: *nvar = n = 5;
				training_set = SSallocate_double_matrix(train_size,n);
				fp = (train_size <= 50) ? fopen("jobshop1.txt","r") : fopen("jobshop2.txt","r");
				for(i=1;i<=train_size;i++) {
					for(j=1;j<=n;++j) 
						fscanf(fp,"%lf",&training_set[i][j]);
					fscanf(fp,"%lf",&train_value[i]);
				}
				break;
	}
	return training_set;
}


double funcion(int np, double *x)
{
	double value=0; 

	switch (np)
	{
		case 1: value = x[1]+x[2];
			    break;
		
		case 2: value = x[1]*x[2];
			    break;

		case 3: value = x[1]/(fabs(x[2])+1);
				break;

		case 4: value = pow(x[1],2)-pow(x[2],3);
				break;

		case 5: value = pow(x[1],3)-pow(x[2],2);
				break;
	}
	return value;
}