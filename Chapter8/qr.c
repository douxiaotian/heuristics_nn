#include <math.h>
#include "ss.h"

void qrdcmp(double **a,int n,double *c,double *d,int *sing)
{
	int i,j,k;
	double scale,sigma,sum,tau;

	*sing=0;
	for(k=1;k<n;k++) {
		scale=0.0;
		for(i=k;i<=n;i++) scale=DMAX(scale,fabs(a[i][k]));
		if(scale==0.0) { // Singular case
			*sing=1;
			c[k]=d[k]=0.0;
		}
		else {
			for(i=k;i<=n;i++) a[i][k] /= scale;
			for(sum=0.0,i=k;i<=n;i++) sum += DSQR(a[i][k]);
			sigma=SIGN(sqrt(sum),a[k][k]);
			a[k][k] += sigma;
			c[k]=sigma*a[k][k];
			d[k] -= scale*sigma;
			for(j=k+1;j<=n;j++) {
				for(sum=0.0,i=k;i<=n;i++) sum += a[i][k]*a[i][j];
				tau=sum/c[k];
				for(i=k;i<=n;i++) a[i][j] -= tau*a[i][k];
			}
		}
	}
	d[n]=a[n][n];
	if(d[n]==0.0) *sing=1;
}

void qrsolv(double **a,int n,double c[],double d[],double b[])
{
	void rsolv(double **a,int n,double d[],double b[]);
	int i,j;
	double sum,tau;

	for(j=1;j<n;j++) {
		for(sum=0.0,i=j;i<=n;i++) sum += a[i][j]*b[i];
		tau =sum/c[j];
		for(i=j;i<=n;i++) b[i] -= tau*a[i][j];
	}
	rsolv(a,n,d,b);
}

void rsolv(double **a,int n,double d[],double b[])
{
	int i,j;
	double sum;

	b[n] /= d[n];
	for(i=n-1;i>=1;i--) {
		for(sum=0.0,j=i+1;j<=n;j++) sum += a[i][j]*b[j];
		b[i]=(b[i]-sum)/d[i];
	}
}
