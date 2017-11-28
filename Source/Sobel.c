/***
 * File: Sobel.c
 */

#include <math.h>
#include "Sobel.h"

static int Sobel_Gx[3][3] = {
   { -1, 0, 1 },
   { -2, 0, 2 },
   { -1, 0, 1 }
};

static int Sobel_Gy[3][3] = {
   {  1,  2,  1 },
   {  0,  0,  0 },
   { -1, -2, -1 }
};

double Sobel_Magnitude(Stencil_t *stencil)
{
   double sumGx = 0, sumGy = 0;

   int i;
   for(i = 0; i < 3; i++)
   {
      sumGx += (Sobel_Gx[0][i] * stencil->top[i])
         +  (Sobel_Gx[1][i] * stencil->middle[i])
         +  (Sobel_Gx[2][i] * stencil->bottom[i]);

      sumGy += (Sobel_Gy[0][i] * stencil->top[i])
         +  (Sobel_Gy[1][i] * stencil->middle[i])
         +  (Sobel_Gy[2][i] * stencil->bottom[i]);
   }

   return sqrt(pow(sumGx, 2) + pow(sumGy, 2));
}
