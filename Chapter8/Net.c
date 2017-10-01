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


extern long EvalNum;  // defined in SS_MAIN.C

Net *InitNet(int n1,int n2,int train_size,double **train_data,double *train_value,
			 int reg,int scale,int activation)
{
	int i,j;
	Net *p;

	p = (Net*) calloc(1,sizeof(Net));
    if(!p) SSabort("Memory allocation problem");

	p->n1			= n1;
	p->n2			= n2;
	p->dim			= ((n1+1)*n2)+n2+1;
	p->regression	= reg;
	p->scale		= scale;
	p->activa		= activation;
	p->train_size	= train_size;
	p->nmax			= 1000;  //500

	// Create and initialize weights
	p->w = SSallocate_double_array(p->dim);
	for(i=1;i<=p->dim;i++) p->w[i] = -0.1+2*(rand()/(double)RAND_MAX); 
	
	// Allocate train data
	p->train_set = SSallocate_double_matrix(train_size,n1);
	for(i=1;i<=train_size;i++)
		for(j=1;j<=n1;j++)	p->train_set[i][j] = train_data[i][j];

	p->train_val = SSallocate_double_array(train_size);
	for(i=1;i<=train_size;i++) p->train_val[i] = train_value[i];

	// Minimum and Maximum value
	p->min_var = SSallocate_double_array(p->n1);
	for(i=1;i<=p->n1;i++) p->min_var[i] = 1E100;
	
	p->max_var = SSallocate_double_array(p->n1);
	for(i=1;i<=p->n1;i++) p->max_var[i] = -1E100;

	p->min_val =  1E100;
	p->max_val = -1E100;
	for(i=1;i<=train_size;i++) 
	{
		if(train_value[i] < p->min_val) p->min_val=train_value[i];
		if(train_value[i] > p->max_val) p->max_val=train_value[i];
		for(j=1;j<=p->n1;j++)
		{
			if(train_data[i][j] < p->min_var[j]) p->min_var[j]=train_data[i][j];
			if(train_data[i][j] > p->max_var[j]) p->max_var[j]=train_data[i][j];
		}
	}

	p->offset = SSallocate_double_array(p->n1);
	p->scalei = SSallocate_double_array(p->n1);
	for(i=1;i<=p->n1;i++) 
	{
		p->offset[i] = (-p->max_var[i]-p->min_var[i] )/(p->max_var[i]-p->min_var[i]);
		p->scalei[i]  = 2 / (p->max_var[i]-p->min_var[i]);
	}

	p->offset_val = (p->max_val*(-0.8)-p->min_val*(0.8) )/(p->max_val-p->min_val);
	p->scale_val  = (0.8-(-0.8) )/(p->max_val-p->min_val);

	return p;
}

double activation(Net *p,double x)
{
	if(p->activa == 1)		return 1/(1+exp(-x));
	else if(p->activa ==2)	return tanh(1.5*x);
	else					return x;
}


double compute_error(Net *p,double *w)
{
	int k,cont,i,j,sig=0,aa;
	double output,error=0,nueva_y,a,value;
	double **red,**matriz,*d,*c,**x,*bb;

	++EvalNum;            // increment the number of evaluations

	// Transfer weights
	red = SSallocate_double_matrix(p->n1+1,p->n2);
	for(cont=1,i=0;i<=p->n1;i++)
	for(j=1;j<=p->n2;j++)
		red[i][j]=w[cont++];

	if(p->regression)
	{
		x	   = SSallocate_double_matrix(p->train_size,p->n2+1);
		matriz = SSallocate_double_matrix(p->n2+1,p->n2+1);
		d	   = SSallocate_double_array(p->n2+1);
		c	   = SSallocate_double_array(p->n2+1);
		bb	   = SSallocate_double_array(p->n2+1);

		// Remaining weights are computed with Regression 
		for(k=1;k<=p->train_size;k++) 
		{	
			x[k][1]=1;
			for(i=1;i<=p->n2;i++)
			{
				a=red[0][i];
				for(j=1;j<=p->n1;j++)
				{
					value = p->train_set[k][j];
					if(p->scale) value=p->offset[j]+value*p->scalei[j];
					a += (red[j][i] * value );
				}
				output = activation(p,a);
				x[k][i+1]=output;
			}
		}	

		// Construct x'x 
		for(i=1;i<=1+p->n2;i++)
		for(j=1;j<=1+p->n2;j++)
		for(aa=1;aa<=p->train_size;aa++)
			matriz[i][j] += x[aa][i]*x[aa][j];

		// Construct x'y 
		for(i=1;i<=p->n2+1;i++)
		for(aa=1;aa<=p->train_size;aa++)
		{
			value = p->train_val[aa];
			if(p->scale) value = p->offset_val+value*p->scale_val;
			bb[i] += x[aa][i]*value;
		}
		qrdcmp(matriz,p->n2+1,c,d,&sig);
		if(sig==0) // x'x^-1 can be computed
		{
			qrsolv(matriz,p->n2+1,c,d,bb);
			for(i=1;i<=p->n2+1;i++)
				w[i+(p->n1+1)*p->n2]=bb[i];

			error=0;
			for(i=1;i<=p->train_size;i++)
			{
				nueva_y=0;
				for(j=1;j<=1+p->n2;j++)
					nueva_y += bb[j]*x[i][j];
				if(p->scale) nueva_y = (nueva_y - p->offset_val) / p->scale_val;
				error += pow(p->train_val[i]-nueva_y,2);
			}
			error /= p->train_size;
		}

		free(c);free(d);free(bb);
		SSfree_double_matrix(matriz,p->n2+1);
		SSfree_double_matrix(x,p->train_size);
	}
	else if(p->regression==0 || sig==1 /* regression failed */ )
	{
		error=0;
		bb=SSallocate_double_array(p->n2);
		for(k=1;k<=p->train_size;k++) 
		{	
			for(i=1;i<=p->n2;i++)
			{
				a=red[0][i];
				for(j=1;j<=p->n1;j++)
				{
					value = p->train_set[k][j];
					if(p->scale) value=p->offset[j]+value*p->scalei[j];
					a += (red[j][i] * value );
				}
				bb[i]=output=activation(p,a);
			}
	
			cont=(p->n1+1)*p->n2+1;
			a=w[cont++];
			for(i=1;i<=p->n2;i++)
				a += bb[i]*w[cont++];

			nueva_y=a;
			if(p->scale) nueva_y = (nueva_y - p->offset_val) / p->scale_val;
			error += pow(p->train_val[k]-nueva_y,2);
		}
		error /= p->train_size;
		free(bb);
	}

	SSfree_double_matrix(red,p->n1+1);
	return error;
}


double net_prediction(Net *p,double *input)
{
	int cont,i,j;
	double output,nueva_y,a;
	double **red,*bb,value;

	// Transfer weights
	red = SSallocate_double_matrix(p->n1+1,p->n2);
	for(cont=1,i=0;i<=p->n1;i++)
	for(j=1;j<=p->n2;j++)
		red[i][j]=p->w[cont++];

	bb=SSallocate_double_array(p->n2);
	for(i=1;i<=p->n2;i++)
	{
		a=red[0][i];
		for(j=1;j<=p->n1;j++)
		{
			value = input[j];
			if(p->scale) value=p->offset[j]+value*p->scalei[j];
			a += (red[j][i] * value );
		}
		bb[i]=output=activation(p,a);
	}
	
	cont=(p->n1+1)*p->n2+1;
	a=p->w[cont++];
	for(i=1;i<=p->n2;i++)
		a += bb[i]*p->w[cont++];
	nueva_y=a;
	if(p->scale) nueva_y = (nueva_y - p->offset_val) / p->scale_val;

	free(bb);
	SSfree_double_matrix(red,p->n1+1);
	return nueva_y;
}