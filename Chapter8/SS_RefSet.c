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

void Intensify(Net *p,int num_iter,double *center,double *besterror)
{
	int j,i=0,total_i=0;
	double error,perturb;

	while(i < num_iter && total_i < num_iter*5)
	{
		++i;++total_i;
		for(j = 1;j <= p->dim; ++j) 
		{
			perturb = center[j]*(-0.05+0.1*(rand()/(double)RAND_MAX));
			p->w[j] = center[j] + perturb;
		}
		error = compute_error(p, p->w);
		SSimprove_solution(p, p->w, &error);
		if (*besterror-error > EPSILON) 
		{
			*besterror = error;
			for(j = 1;j <= p->dim; ++j) 
				center[j] = p->w[j];
			i=0;
		}
	}
}


void Initiate_RefSet(Net *p,SS *prob)
{
	double *current,*min_dist,*value,**solutions;
	int j,i,k,*index,*index2,cont=0;
	double current_value,perturb;

	prob->iter=1;
	prob->new_elements = 1;

	current = SSallocate_double_array(prob->n_var);
	if(!current) SSabort("Memory allocation problem");

	min_dist = SSallocate_double_array(prob->PSize);	
	if(!min_dist) SSabort("Memory allocation problem");

	value = SSallocate_double_array(prob->PSize);	
	if(!min_dist) SSabort("Memory allocation problem");

	solutions = SSallocate_double_matrix(prob->PSize,prob->n_var);
	if(!solutions) SSabort("Memory allocation problem");

	for(i=1;i<=prob->PSize;i++)
	{
		/* Generate new solution */
		for(j=1;j<=prob->n_var;j++)	
			current[j] = SSGenerate_value(prob,j);

		/* Evaluate Solution */
		current_value=compute_error(p,current);
		if(is_new(prob,solutions,i-1,current))
		{
			/* Store solution in matrix "solutions" */
			for(j=1;j<=prob->n_var;j++)
				solutions[i][j] = current[j];
		
			value[i]=current_value;
		}
		else {i--;cont++;}

		if(cont>prob->PSize/2) {
			prob->digits++;
			cont=0;
		}
	}
	index = orden_indices(value,prob->PSize,-1);

	/* Add the best b1 to RefSet1 */
	for(i=1;i<=prob->b1 / 2;i++)
	{
		for(j=1;j<=prob->n_var;j++)
			prob->RefSet1[i][j] = solutions[index[i]][j];
		
		prob->value1[i] = value[index[i]];
		
		if(prob->LS) 
			SSimprove_solution(p,prob->RefSet1[i],&(prob->value1[i]));

		prob->order1[i] = i;
		prob->iter1[i]  = 1;
	}


	/*Add the second b2 to RefSet2 */
	for(i=1;i<=prob->b1/2;i++)
	{
		k = i+(prob->b1/2);
		for(j=1;j<=prob->n_var;j++)
		{
			perturb = prob->RefSet1[i][j]*(-0.3+0.6*(rand()/(double)RAND_MAX));
			prob->RefSet1[k][j] = prob->RefSet1[i][j] + perturb;
		
			prob->value1[k]= compute_error(p,prob->RefSet1[k]);
			prob->iter1[k]  = 1;
		}
	}

	index2 = orden_indices(prob->value1,prob->b1,-1);
	
	for(i=1;i<=prob->b1;i++)
		prob->order1[i] = index2[i];

	free(index);free(index2);free(current);free(min_dist);
	free(value);SSfree_double_matrix(solutions,prob->PSize);
}


void Update_RefSet(Net *p,SS *prob)
{
	double *current,*min_dist,*value,**solutions;
	int j,i,k,*index2;
	double perturb;

	prob->iter++;
	prob->digits++;

	current = SSallocate_double_array(prob->n_var);
	if(!current) SSabort("Memory allocation problem");

	min_dist = SSallocate_double_array(prob->PSize);	
	if(!min_dist) SSabort("Memory allocation problem");

	value = SSallocate_double_array(prob->PSize);	
	if(!min_dist) SSabort("Memory allocation problem");

	solutions = SSallocate_double_matrix(prob->PSize,prob->n_var);
	if(!solutions) SSabort("Memory allocation problem");


	/* Improve First elements */
	if(prob->LS ){
		prob->ImpCount=0;
		for(i=1;i<=prob->b1/2;i++)
			SSimprove_solution(p,prob->RefSet1[prob->order1[i]],
							 &(prob->value1[prob->order1[i]]));
	}
	
	/*Add the second b2 to RefSet2 */
	for(i=1;i<=prob->b1/2;i++)
	{
		k = prob->order1[i+(prob->b1/2)];
		for(j=1;j<=prob->n_var;j++)
		{
			perturb = prob->RefSet1[i][j]*(-0.01+0.02*(rand()/(double)RAND_MAX));
			if(fabs(perturb) <  EPSILON) perturb = -0.1+0.2*(rand()/(double)RAND_MAX);
			prob->RefSet1[k][j] = prob->RefSet1[i][j] + perturb;
		}
		prob->value1[k]= compute_error(p,prob->RefSet1[k]);

		if(prob->LS) SSimprove_solution(p,prob->RefSet1[k], &(prob->value1[k]));
		prob->iter1[k]  = 1;
	}

	index2 = orden_indices(prob->value1,prob->b1,-1);
	
	for(i=1;i<=prob->b1;i++)
	{
		prob->order1[i] = index2[i];
		prob->iter1[i]  = prob->iter;
	}

	prob->new_elements = 1;

	free(index2);free(current);free(min_dist);free(value);
	SSfree_double_matrix(solutions,prob->PSize);
}

void Combine_RefSet(Net *p,SS *prob)
{
	int i,j,a,s,pull_size,total_size,number,*index;
	double **offsprings,**pull,*value;

	prob->new_elements=0;
	offsprings = SSallocate_double_matrix(4,prob->n_var);

	/* New solutions are temporarily stored in a pull */
	pull_size=0;
	total_size=(4*prob->b1*prob->b1);
	pull = SSallocate_double_matrix(total_size,prob->n_var);
	value = SSallocate_double_array(total_size);

	/* Combine elements in RefSet1 */
	for(i=1;i<prob->b1;i++)
	for(j=i+1;j<=prob->b1;j++)
	{
		/* Combine solutions not combined in the past */
		if(prob->iter1[i]>prob->last_combine ||
		   prob->iter1[j]>prob->last_combine   )
		{
			if(i<= prob->b1/2.0 && j<= prob->b1/2.0) number = 4;
			else if( i<= prob->b1/2.0 )				 number = 3;
			else									 number = 2;

			SScombine(prob,prob->RefSet1[i],prob->RefSet1[j],offsprings,number);

			for(a=1;a<=number;a++)
			{
				pull_size++;
				for(s=1;s<=prob->n_var;s++)
					pull[pull_size][s]=offsprings[a][s];
			}
		}
	}


	/* Update, if necessary, Reference Set */
	
	prob->last_combine=prob->iter;
	prob->iter++;

	for(a=1;a<=pull_size;a++)
		value[a]=compute_error(p,pull[a]);

	index = orden_indices(value,pull_size,-1);

	for(a=1;a<=prob->b1;a++) 
		try_add_RefSet1(p,prob,pull[index[a]],value[index[a]]);

	SSfree_double_matrix(pull,total_size);
	SSfree_double_matrix(offsprings,4);
	free(value);
	free(index);
	
}

double SSGenerate_value(SS *prob,int a)
{
	int i,j;
	int *rfrec; /* reverse frec to penalize high frecs. */
	double r,value,low,range;
	int *frec;

	frec = prob->ranges[a];
	low  = prob->low[a];
	range= prob->high[a]-prob->low[a];

	rfrec = SSallocate_int_array(5);
	if(!rfrec) SSabort("Problems allocating memory");

	for(i=1;i<=4;i++)
	{
		rfrec[i]  = frec[0] - frec[i];
		rfrec[0] += rfrec[i];
	}

	if(rfrec[0]==0)
		i = getrandom(1,4);
	else
	{
		/* Select a subrange (from 1 to 4) according to rfrec */
		j = getrandom(1,rfrec[0]);
		i=1;
		while(j>rfrec[i])
			j -= rfrec[i++];
	}
	if(i>4) SSabort("Problems generating values");;

	/* i is the selected subrange */
	frec[i]++;
	frec[0]++;
	free(rfrec);

	/* Randomly select an element in subrange i */
	r = rand()/(float)RAND_MAX;
	value=low+(i-1)*(range/4) + (r*range/4);
	return value;
}

