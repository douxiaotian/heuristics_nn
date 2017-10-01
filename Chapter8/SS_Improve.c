/***********************************
 *								   *
 *	Scatter Search C code		   *
 *	Version: 2.0				   *
 *	Authors: M. Laguna & R. Marti  *
 *	Copyright © 2002			   *
 *								   *
 ***********************************/

#include "ss.h"

void SSimprove_solution(Net *net,double *x,double *value)
{
	double **p,*y,perturb, xold;
	int i,j,nvar;

	nvar = net->dim;

	p = SSallocate_double_matrix(nvar+1,nvar);
	y = SSallocate_double_array(nvar+1);

	for(i=1;i<=nvar;i++) p[1][i]=x[i];
	y[1]=*value;

	for(i=1;i<=nvar;i++)
		for(j=1;j<=nvar;j++)
			p[i+1][j]=x[j];
	
	for(j=1;j<=nvar;j++)
	{
		xold = x[j];
		perturb = -1+2*(rand()/(double)RAND_MAX);
		x[j] += perturb;
		p[j+1][j] = x[j];
		y[j+1] = compute_error(net,x);
		x[j] = xold;
	}

	/* Call Nelder and Mead's Simplex method */
	SS_Simplex(p,y,nvar,net->nmax,net);

	for(i=1;i<=nvar+1;i++)
	{
		if( *value>y[i])
		{
			for(j=1;j<=nvar;j++)
				x[j]=p[i][j];
			*value=y[i];
		}
	}
	free(y);
	SSfree_double_matrix(p,nvar+1);
}




void SS_Simplex(double **simplex,double *values,int nvar,int max_eval,Net *net)
{
	int i,j,best_index,worst_index,nextworst_index,eval_num;
	double factor,new_value, move_value,sum,*cum_simplex;

	cum_simplex = SSallocate_double_array(nvar+1);
	eval_num=0;factor=1.0;

	/* Add solutions */
	for (j=1;j<=nvar;j++)
	{ 
		sum=0.0;
		for (i=1;i<=nvar+1;i++)
			sum += simplex[i][j];
		cum_simplex[j]=sum;
	}

	while(eval_num < max_eval)
	{
		/* Compute best, worst and next worst points */
		best_index = 1;
		if(values[1] > values[2])
		{
			nextworst_index = 2;
			worst_index =1;
		}
		else
		{
			nextworst_index = 1;
			worst_index =2;
		}


		for (i = 1; i <= nvar+1; i++)
		{
			if (values[i] < values[best_index])
				best_index = i;
			if (values[i] > values[worst_index])
			{
				nextworst_index = worst_index;
				worst_index = i;
			}
			else if (values[i] > values[nextworst_index] && i != worst_index)
				nextworst_index = i;
		}

		new_value = SSMove(nvar,simplex[worst_index],&(values[worst_index]),cum_simplex,-factor,net);
		eval_num++;

		if (new_value <= values[best_index])
		{
			new_value = SSMove(nvar,simplex[worst_index],&(values[worst_index]),cum_simplex,2*factor,net);
			eval_num++;
		}
		else if (new_value >= values[nextworst_index])
		{
			move_value = values[worst_index];
			new_value  = SSMove(nvar,simplex[worst_index],&(values[worst_index]),cum_simplex,factor/2,net);
			eval_num++;

			if (new_value >= move_value)
			{
				for (i = 1; i <= nvar+1; i++)
					if (i != best_index)
					{
						for (j = 1; j <= nvar; j++)
						{
							cum_simplex[j] = 0.5*(simplex[i][j]+simplex[best_index][j]);
							simplex[i][j] = cum_simplex[j];
						}
						values[i] = compute_error(net,cum_simplex);
						eval_num++;
					}
							
				/* Add solutions */
				for (j=1;j<=nvar;j++)
				{ 
					for (i=1, sum=0.0;i<=nvar+1;i++)
						sum += simplex[i][j];
					cum_simplex[j]=sum;
				}
			}
		}
	}
	free(cum_simplex);
}


double SSMove(int nvar,double *worst_point,double *worst_value, double *cum_simplex, double factor,Net *net)
{
	int i;
	double rfac,*new_point,new_value;

	/* Generate a new point */
	new_point = SSallocate_double_array(nvar+1);
	rfac = (1.0-factor)/(double)nvar;
	for (i=1;i<=nvar;i++)
		new_point[i]=cum_simplex[i]*rfac - worst_point[i]*(rfac-factor);
	new_value = compute_error(net,new_point);

	/* Check worst_point replacement */
	if (new_value < *worst_value)
	{
		*worst_value = new_value;
		for (i=1;i<=nvar;i++)
		{
			cum_simplex[i] += new_point[i]-worst_point[i];
			worst_point[i] = new_point[i];
		}
	}
	free(new_point);
	return new_value;
}

