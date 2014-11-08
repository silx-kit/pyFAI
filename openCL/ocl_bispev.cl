/*
 *   Project: Distortion correction based on spline for PyFAI.
 *
 *   Copyright (C) 2013-2014 SESAME, P.O. Box 7, Allan 19252, Jordan
 *
 *   Principal authors: Zubair Nawaz <zubair.nawaz@gmail.com>
 *   Last revision: 20/10/2014
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//static const int WORK_SIZE = 256;

#include "for_eclipse.h"


/**
 * \brief  
 * 
 * subroutine fpbspl evaluates the (k+1) non-zero b-splines of
 * degree k at t(l) <= x < t(l+1) using the stable recurrence
 * relation of de boor and cox.
 * 
 *
 * @param d_t:  Pointer to global memory with the data in int
 * @param n:    size of d_t
 * @param k:    size of d_t
 * 
 * @param h:    output: array of floats
 */


__kernel__ void
fpbspl(float* d_t, int n, int k, float x, int l, float* h)      {
    h[0] = 1;
    float hh[5];

    for( int j=1; j<=k; j++)    {
        for( int i=1; i<=j; i++)
            hh[i-1] = h[i-1];

        h[0] = 0;
        for ( int i=1; i<=j; i++)       {
            int li = l + i;
            int lj = li - j;
            float f = hh[i-1]/(d_t[li-1]-d_t[lj-1]);
            h[i-1] = h [i-1] + f * (d_t[li-1] - x);
            h[i] = f * (x - d_t[lj-1]);
        }
    }
}


/* parallel version of fpbisp contains 2 parts, first is inherently serial, therefore its called
 * fpbisp_serial1 and other part is parallel called fpbisp_parallel2.
 * One possibility is that serial part be written in Cython, then wx and wy have to transfered to
 * the parallel part
 * 
 * d_tx: array of float size nx containing position of knots in x
 * d_ty: array of float size ny containing position of knots in y
 * kx, ky: spline order (often  3)    
 * d_x, d_y : array of float of size mx, my specifying the domain over which to evaluate the spline
 * d_wx, d_wy: scratch space 
 * 
 */
__kernel void
fpbisp_serial1(	__global float* d_tx, int nx,
				__global  float* d_ty, int ny, 
				int kx,	int ky,
				__global float* d_x, int mx,
				__global float* d_y, int my,
				__global float* d_wx,
				__global float* d_wy,
				__global int* d_lx,
				__global int* d_ly)	{

	int kx1 = kx+1;
	int nkx1 = nx - kx1;

	float tb = d_tx[kx1-1];	// adding -1 in the index
	float te = d_tx[nkx1];	// adding -1 in the index

	int l = kx1;
	int l1 = l + 1;

	int ky1 = ky + 1;
	int nky1 = ny - ky1;
	float h[6];
	
//	printf("Inside fpbisp_serial1");	

	for (int i=1; i<=mx; i++)	{
		int arg = d_x[i-1];

		if (arg < tb)
			arg = tb;
		else if (arg > te)
			arg = te;

		while ( !( (arg < d_tx[l1-1]) || (l == nkx1) ) )	{
			l = l1;
			l1 = l+1;
		}

		fpbspl_serial(d_tx, nx, kx, arg, l, h);

		d_lx[i-1] = l - kx1;

		for (int j=1; j<=kx1; j++)	{
			d_wx[(i-1)*kx1 + (j-1)] = h[j-1]; // wx[i-1,j-1]=h[j-1]
			//printf("wx[i-1,j-1] = %f \n", h[j-1]);
		}
	}


	tb = d_ty[ky1-1];
	te = d_ty[nky1];

	l = ky1;
	l1 = l + 1;

	for (int i=1; i<=my; i++)	{
		int arg = d_y[i-1];

		if (arg < tb)
			arg = tb;
		else if (arg > te)
			arg = te;

		while ( !( (arg < d_ty[l1-1]) || (l == nky1) ) )	{
			l = l1;
			l1 = l+1;
		}

		fpbspl_serial(d_ty, ny, ky, arg, l, h);

		d_ly[i-1] = l - ky1;

		for (int j=1; j<=ky1; j++)
			d_wy[(i-1)*ky1 + (j-1)] = h[j-1]; // wy[i-1,j-1]=h[j-1]

	}

}

/*
 * Second part of parallel fpbisp
 */
__kernel void
fpbisp_parallel2(int kx, int ky, int mx, int my, int ny,__global float* d_c,__global float* d_wx,__global float* d_wy,
			__global int* d_lx,__global int* d_ly,__global float* d_z)	{

	float hi[4];	// keep local values of hi for every thread
	float hj[4];	// keep local values of hi for every thread
	int kx1 = kx + 1;
	int ky1 = ky + 1;
	int nky1 = ny - ky1;
	float h_i;

	int id = get_global_id(0) + 1;

	// exits all the threads whose id is greater than equal to mx
	if (id > my)
		return;
	
	float tmp;
	int pm = 0;		// previous value of m
		
	// every thread has a private copy of hj, this way it reduces the memory cost	
	for (int j1=1; j1<=ky1; j1++)
		hj[j1-1] = d_wy[(id-1)*ky1 + (j1-1)];


	for (int i=1; i <=mx; i++)	{	// each iteration of i computes a row in z
		for (int i1=1; i1<=kx1; i1++)
			hi[i1-1] = d_wx[(i-1)*kx1 + (i1-1)];	// hi[i1-1] = wx[i-1,i1-1]

		//int l = d_lx[i-1] * nky1;

		int l1 = d_lx[i-1] * nky1 + d_ly[id-1];
		float sp = 0;
		float err = 0;
		for (int i1=1; i1<=kx1; i1++)	{
			int l2 = l1;
			h_i = hi[i1-1];
			for (int j1=1; j1<=ky1; j1++)	{
				l2 = l2 + 1;
				float a = d_c[l2-1] * h_i * hj[j1-1] - err;
				tmp = sp + a;
				err = (tmp - sp) - a;
				sp = tmp;
				//sp = sp + d_c[l2-1] * h_i * hj[j1-1]; // sp = sp + c[l2-1] * hi[i1-1] * wy[j-1, j1-1]
			}
			l1 = l1 + nky1;
		}

		int m = pm + id - 1;
		d_z[m] = sp;
		pm = i * my;	// updates the pm for the next row

	}

}
