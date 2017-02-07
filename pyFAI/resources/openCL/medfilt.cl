/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Median filter for 1D, 2D and 3D datasets, only 2D for now
 *
 *
 *   Copyright (C) 2017-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 07/02/2017
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

/*
 * Needs to be concatenated with btonic.cl prior to compilation
*/

/*
 * Function to read at a given position in a float8 based on switches,
 * Probably inefficient
 *
 */

static float read_float8(float8 vect,
                         size_t index)
{
    float value = 0.0f;
    switch (index)
    {
        case 0:
            value = vect.s0;
        case 1:
            value = vect.s1;
        case 2:
            value = vect.s2;
        case 3:
            value = vect.s3;
        case 4:
            value = vect.s4;
        case 5:
            value = vect.s5;
        case 6:
            value = vect.s6;
        case 7:
            value = vect.s7;
    }
    return value;

}

/*
 * Function to write at a given position in a float8 based on switches,
 * Probably inefficient
 *
 */
static float8 write_float8(float8 vect,
                         size_t index,
                         float value)
{
    switch (index)
    {
        case 0:
            vect.s0 = value;
        case 1:
            vect.s1 = value;
        case 2:
            vect.s2 = value;
        case 3:
            vect.s3 = value;
        case 4:
            vect.s4 = value;
        case 5:
            vect.s5 = value;
        case 6:
            vect.s6 = value;
        case 7:
            vect.s7 = value;
    }
    return vect;
}


/*
 *  Perform the 2D median filtering of an image 2D image
 *
 * dim0 => wg=number_of_element in the tile /8
 * dim1 = y: wg=1
 * dim2 = x: wg=1
 *
 * Actually the workgoup size is a bit more complicated:
 * if window size = 1,3,5,7: WG1=8
 * if window size = 9,11,13,15: WG1=32
 * if window size = 9,11,13,15: WG1=64
 *
 * More Generally the workgroup size must be: at least: kfs2*(kfs1+7)/8
 * Each thread treats 8 values aligned vertically, this allows (almost)
 * coalesced reading between threads in one line of the tile.
 *
 * Later on it will be more efficient to re-use the same tile
 * and slide it vertically by one line.
 * The additionnal need for shared memory will be kfs2 floats and a float8 as register.
 *
 * Theoritically, it should be possible to handle up to windows-size 45x45
 */
__kernel void medfilt2d(__global float *image,  // input image
                        __global float *result, // output array
                        __local  float4 *l_data,// local storage 4x the number of threads
                                 int khs1,      // Kernel half-size along dim1 (nb lines)
                                 int khs2,      // Kernel half-size along dim2 (nb columns)
                                 int height,    // Image size along dim1 (nb lines)
                                 int width)     // Image size along dim2 (nb columns)
{
    int threadid = get_local_id(0);
    int wg = get_local_size(0);
    int x = get_global_id(1);

    if (x < width)
    {
        union
        {
            float  ary[8];
            float8 vec;
        } output, input, storage;
        input.vec = (float8)(MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT);
        int kfs1 = 2 * khs1 + 1; //definition of kernel full size
        int kfs2 = 2 * khs2 + 1;
        int nbands = (kfs1 + 7) / 8; // 8 elements per thread, aligned vertically in 1 column
        for (int y=0; y<height; y++)
        {
            //Select only the active threads, some may remain inactive
            int nb_threads =  (nbands * 8 * kfs2);
            int band_nr = threadid / kfs2;
            int band_id = threadid % kfs2;
            int pos_x = clamp((int)(x + band_id - khs2), (int) 0, (int) width-1);
            if (y == 0)
            {
                int max_vec = ((band_nr + 1) == nbands)? kfs1 % 8 : 8;
                for (int i=0; i<max_vec; i++)
                {
                    if (threadid<nb_threads)
                    {
                        int pos_y = clamp((int)(y + 8 * band_nr + i - khs1), (int) 0, (int) height-1);
                        input.ary[i] = image[pos_x + width * pos_y];
                    }
                }
            }
            else
            {
                //Offset to the bottom
                input.vec.s7 = MAXFLOAT;
                input.vec.s6 = storage.vec.s7;
                input.vec.s5 = storage.vec.s6;
                input.vec.s4 = storage.vec.s5;
                input.vec.s3 = storage.vec.s4;
                input.vec.s2 = storage.vec.s3;
                input.vec.s1 = storage.vec.s2;
                input.vec.s0 = storage.vec.s1;
                //storage.s0 is lost as we move down

/*works*/
                int max_vec = ((band_nr + 1) == nbands)? kfs1 % 8 : 8;
                if (threadid<nb_threads)
                {
                    int i = max_vec - 1;
                    int pos_y = clamp((int)(y + 8 * band_nr + i - khs1), (int) 0, (int) height-1);
                    input.ary[i] = image[pos_x + width * pos_y];
                }
/* Does not work!
                //store storage.s0 to some shared memory to retieve it from another thread.
                l_data[threadid].x = storage.vec.s0;
                barrier(CLK_LOCAL_MEM_FENCE);
                int read_from = threadid + kfs2;
                float value;
                if (read_from<nb_threads)
                {
                    input.vec.s7 = l_data[read_from].x;
                }
                if ((band_nr + 1) == nbands)  //we are on the last line
                //else if (threadid<nb_threads) //we are on the last line
                {
                    int pos_y = clamp((int)(y + khs1), (int) 0, (int) height-1);
                    //Coalesced read in global memory
                    input.ary[kfs1 % 8] = image[pos_x + width * pos_y];
                }
                */


            }
            //store a copy of the input
            storage = input;

            //This function is definied in bitonic.cl
            output.vec = my_sort_file(get_local_id(0), get_group_id(0), get_local_size(0),
                                       input.vec, l_data);

            size_t target = (kfs1 * kfs2) / 2;
            if (threadid == (target / 8))
            {
                result[y * width + x] = output.ary[target % 8];
            }

        }
    }
}

