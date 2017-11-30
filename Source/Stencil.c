/***
 * File: Stencil.h
 */

#include "Stencil.h"

void Stencil_MoveRight(Stencil_t *stencil)
{
   stencil->top++;
   stencil->middle++;
   stencil->bottom++;
}

void Stencil_MoveToNextRow(Stencil_t *stencil)
{
   /*
    * Since the array is contiguous and the stencil is just past the rightmost
    * boundary, the next row wraps around and is two elements ahead of
    * the current element
    */
   stencil->top += 2;
   stencil->middle += 2;
   stencil->bottom += 2;
}
