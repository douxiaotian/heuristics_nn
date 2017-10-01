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

void try_add_RefSet1(Net *p,SS *prob,double *sol,double value)
{
	int i,j,worst_index;
	double worst_value;

	//value=compute_error(p,sol);
	if(prob->LS) SSimprove_solution(p,sol,&value);

	worst_index=prob->order1[prob->b1];
	worst_value=prob->value1[worst_index];

	if(is_new(prob,prob->RefSet1,prob->b1,sol) && value<worst_value)   
	{
		i=prob->b1;
		while(i>=1 && value<prob->value1[prob->order1[i]])
			i--;
		i++;

		/* Replace solution */
		for(j=1;j<=prob->n_var;j++)
			prob->RefSet1[worst_index][j]=sol[j];
		prob->value1[worst_index]=value;
		prob->iter1[worst_index]=prob->iter;

		/* Update Order */
		for(j=prob->b1;j>i;j--)
			prob->order1[j]=prob->order1[j-1];
		
		prob->order1[i]=worst_index;
		prob->new_elements=1;

	}
}
	

void SScombine(SS *prob,double *x,double *y,double **offsprings,int number)
{
	int j;
	double a,*d,r;

	d=SSallocate_double_array(prob->n_var);
	if(!d) SSabort("Memory allocation problem");

	r = rand()/(float)RAND_MAX;
	for(j=1;j<=prob->n_var;j++)
		d[j] = (y[j] - x[j]) / 2;
	
	/* Generate C2 */
	for(j=1;j<=prob->n_var;j++)
		offsprings[1][j] = x[j] + r*d[j];

	if(number>=2) /* Generate C1 or C3 */
	{
		a = rand()/(float)RAND_MAX;
		for(j=1;j<=prob->n_var;j++)
		{
			if(a<=0.5)	offsprings[2][j] = x[j] - r*d[j];
			else		offsprings[2][j] = y[j] + r*d[j];
		}
	}

	if(number>=3) /* Generate the other one (C1 or C3) */
	{
		a = rand()/(float)RAND_MAX;
		for(j=1;j<=prob->n_var;j++)
		{
			if(a>0.5)	offsprings[3][j] = x[j] - r*d[j];
			else		offsprings[3][j] = y[j] + r*d[j];
		}
	}

	if(number==4) /* Generate another C2 */
	{
		r = rand()/(float)RAND_MAX;
		for(j=1;j<=prob->n_var;j++)
			offsprings[4][j] = x[j] + r*d[j];
	}
	free(d);
}


int is_new(SS *prob,double **solutions,int dim,double *sol)
{
	int i,j,is_new;
	double precision=0;

	precision=1/pow(10,prob->digits);

	for(i=1;i<=dim;i++)
	{
		is_new=0;
		for(j=1;j<=prob->n_var;j++)
			if(fabs(solutions[i][j] - sol[j]) >= precision)
				is_new=1;
		if(is_new==0) return 0;
	}
	return 1;
}


double distance_to_RefSet1(SS *prob,double *sol)
{
	double d,min_dist=DBL_MAX;
	int a,j;

	for(a=1;a<=prob->b1/2;a++)
	{
		d=0;
		for(j=1;j<=prob->n_var;j++)
			d += pow(sol[j]-prob->RefSet1[a][j],2);
		if(min_dist> d)
			min_dist=d;
	}

	return min_dist;
}


int *orden_indices(double *pesos,int num,int tipo)
{
	int *indices,b,t,j,i,tempi;
	double temp,*coste;
	
	coste=SSallocate_double_array(num);
	indices=SSallocate_int_array(num);

	for(i=1;i<=num;i++)
	{
		coste[i]=pesos[i];
		indices[i]=i;
	}
	
	b=num;
	while(b!=0)
	{   
		t=0;              
		for(j=1;j<=b-1;j++)
		{
			if( (tipo==1  && coste[j]<coste[j+1]) ||
				(tipo==-1 && coste[j]>coste[j+1])    )
			{
				temp=coste[j+1];
				coste[j+1]=coste[j];
				coste[j]=temp;
				
				tempi=indices[j+1];
				indices[j+1]=indices[j];
				indices[j]=tempi;
				
				t=j;
			}		
		}
		b=t;
	}	
	free(coste);
	return indices;
}

void SSabort(char *texto)
{
	printf("%s",texto);
	exit(6);
}