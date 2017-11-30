/***
 * File: maxwell_griffin_lab5.c
 * Desc: Performs a Sobel edge detection operation on a .bmp using MPI for
 *       parallelization.
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

extern "C"
{
#include "read_bmp.h"
#include "Sobel.h"
}

#define PIXEL_BLACK (0)
#define PIXEL_WHITE (255)
#define PERCENT_BLACK_THRESHOLD (0.75)

#define CUDA_GRIDS (1)
#define CUDA_BLOCKS_PER_GRID (32)
#define CUDA_THREADS_PER_BLOCK (128)

#define MS_PER_SEC (1000)
#define NS_PER_MS (1000 * 1000)
#define NS_PER_SEC (NS_PER_MS * MS_PER_SEC)

#define LINEARIZE(row, col, dim) \
   (((row) * (dim)) + (col))

static struct timespec rtcStart;
static struct timespec rtcEnd;

/*
 * Display all header and image information.
 *
 * @param inputFile -- name of the input image
 * @param outputFile -- name of the serial output image
 * @param imageHeight -- in pixels
 * @param imageWidth -- in pixels
 */
void DisplayParameters(
   char *inputFile,
   char *outputFile,
   int imageHeight,
   int imageWidth)
{
   printf("********************************************************************************\n");
   printf("lab5: Sobel edge detection using MPI.\n");
   printf("\n");
   printf("Input image: %s \t(Height: %d pixels, width: %d pixels)\n", inputFile, imageHeight, imageWidth);
   printf("Output image: \t%s\n", outputFile);
   printf("\n");
}

/*
 * Display the MPI paramaters and timing and convergence results to the screen.
 *
 * @param convergenceThreshold
 */
void DisplayResults(convergenceThreshold)
{
   int communicatorSize;
   MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

   double executionTime = (LINEARIZE(rtcEnd.tv_sec, rtcEnd.tv_nsec, NS_PER_SEC)
   - LINEARIZE(rtcStart.tv_sec, rtcStart.tv_nsec, NS_PER_SEC))
   / ((double)NS_PER_SEC);

   printf("MPI communicator size:\n", communicatorSize);
   printf("Time taken for MPI operation: %lf\n", executionTime);
   printf("Convergence Threshold: %d\n", convergenceThreshold);
   printf("********************************************************************************\n");
}

/*
 * Using MPI parallelization, perform Sobel edge detections on an input pixel
 * buffer at increasing brightness thresholds until convergence, i.e. 75% of
 * pixels in the output pixel buffer are black.
 *
 * @param input -- input pixel buffer
 * @param output -- output pixel buffer
 * @param height -- height of pixel image
 * @param width -- width of pixel image
 * @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD pixels are black
 */
int SerialSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width)
{
   int gradientThreshold, blackPixelCount = 0;
   for(gradientThreshold = 0; blackPixelCount < (height * width * 3 / 4); gradientThreshold++)
   {
      blackPixelCount = 0;

      // Initialize stencil to sit centered at input[1][1]
      Stencil_t pixel = {
         .top =    &input[LINEARIZE(0, 0, width)],
         .middle = &input[LINEARIZE(1, 0, width)],
         .bottom = &input[LINEARIZE(2, 0, width)]
      };

      // Skip first and last row (to avoid top/bottom boundaries)
      for(int row = 1; row < (height - 1); row++)
      {
         // Skip first and last column (to avoid left/right boundaries)
         for(int col = 1; col < (width - 1); col++)
         {
            if(Sobel_Magnitude(&pixel) > gradientThreshold)
            {
               output[LINEARIZE(row, col, width)] = PIXEL_WHITE;
            }
            else
            {
               output[LINEARIZE(row, col, width)] = PIXEL_BLACK;
               blackPixelCount++;
            }

            Stencil_MoveRight(&pixel);
         }

         Stencil_MoveToNextRow(&pixel);
      }
   }

   return gradientThreshold;
}

/*
* Main function.
*/
int main(int argc, char *argv[])
{
   // Call MPI_Init to setup the MPI environment and shift out all the MPI-specific command line args
   MPI_Init(&argc, &argv);

   // Check for correct number of remaining command line args
   if (argc != 4)
   {
      printf("Error: Incorrect arguments: <input.bmp> <serial_output.bmp> <cuda_output.bmp>\n");
      return 0;
   }

   // Open the files specified by the command line args
   FILE *inputFile = fopen(argv[1], "rb");
   FILE *serialOutputFile = fopen(argv[2], "wb");
   FILE *cudaOutputFile = fopen(argv[3], "wb");
   if(inputFile == NULL)
   {
      printf("Error: %s could not be opened for reading.", argv[1]);
   }

   // Read in input image and allocate space for new output image buffers
   uint8_t *inputImage = (uint8_t *)read_bmp_file(inputFile);
   uint8_t *serialOutputImage = (uint8_t *)malloc(get_num_pixel());
   uint8_t *cudaOutputImage = (uint8_t *)malloc(get_num_pixel());

   DisplayParameters(argv[1], argv[2], argv[3], get_image_height(), get_image_width());

   printf("Performing serial Sobel edge detection.\n");
   clock_gettime(CLOCK_REALTIME, &rtcSerialStart);
   int serialConvergenceThreshold = SerialSobelEdgeDetection(inputImage, serialOutputImage, get_image_height(), get_image_width());
   clock_gettime(CLOCK_REALTIME, &rtcSerialEnd);

   printf("Performing CUDA parallel Sobel edge detection.\n");
   clock_gettime(CLOCK_REALTIME, &rtcParallelStart);
   int parallelConvergenceThreshold = ParallelSobelEdgeDetection(inputImage, cudaOutputImage, get_image_height(), get_image_width());
   clock_gettime(CLOCK_REALTIME, &rtcParallelEnd);

   DisplayResults(serialConvergenceThreshold, parallelConvergenceThreshold);

   // Write output image buffers. Closes files and frees buffers.
   write_bmp_file(serialOutputFile, serialOutputImage);
   write_bmp_file(cudaOutputFile, cudaOutputImage);

   MPI_Finalize();
}
