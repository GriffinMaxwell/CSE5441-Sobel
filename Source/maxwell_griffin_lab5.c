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

#define NS_PER_SEC (1000 * 1000 * 1000)

#define LINEARIZE(row, col, dim) \
   (((row) * (dim)) + (col))

/*
 * Find the difference in seconds between two struct timespecs
 *
 * @param start -- the earlier time
 * @param end -- the later time
 */
static double CalculateExecutionTime(struct timespec start, struct timespec end)
{
   return (LINEARIZE(rtcEnd.tv_sec, rtcEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcStart.tv_sec, rtcStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC);
}

/*
 * Display all header and image information.
 *
 * @param inputFile -- name of the input image
 * @param outputFile -- name of the serial output image
 * @param imageHeight -- in pixels
 * @param imageWidth -- in pixels
 */
static void DisplayParameters(
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
 * @param communicatorSize
 * @param executionTime
 * @param convergenceThreshold
 */
static void DisplayResults(
   int communicatorSize,
   double executionTime,
   int convergenceThreshold)
{
   printf("MPI communicator size: %d\n", communicatorSize);
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
static int SerialSobelEdgeDetection(
   uint8_t *input,
   uint8_t *output,
   int height,
   int width)
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

   struct timespec rtcStart;
   struct timespec rtcEnd;

   int communicatorSize, myRank;
   MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   // Define the "master process" as the process with the largest rank
   // Rank is zero indexed, so largest rank is communicator size - 1
   const int masterProcessRank = communicatorSize - 1;

   // Check for correct number of remaining command line args
   if (argc != 4)
   {
      printf("Error: Incorrect arguments: <input.bmp> <serial_output.bmp> <cuda_output.bmp>\n");
      return 0;
   }

   // Open the input file specified by the command line arg
   FILE *inputFile = fopen(argv[1], "rb");
   if(inputFile == NULL)
   {
      printf("Error: %s could not be opened for reading.", argv[1]);
   }

   // Buffers to hold the entire input image and whatever block of the output
   // image buffer this process is processing
   uint8_t *inputImage, *outputImageBlock;

   // Read input image, determine block size, and allocate output image buffer
   inputImage = (uint8_t *)read_bmp_file(inputFile);

   // Divvy out blocks of pixels to each process.
   // The master process gets the extra pixels from uneven division
   int slaveBlockSize = get_num_pixel() / communicatorSize;
   int numExtraPixels = get_num_pixel() % communicatorSize;
   int myBlockSize = (myRank == masterProcessRank) ? slaveBlockSize + numExtraPixels : slaveBlockSize

   outputImageBlock = (uint8_t *)malloc(myBlockSize);

   if(myRank == masterProcessRank)
   {
      DisplayParameters(argv[1], argv[2], argv[3], get_image_height(), get_image_width());

      printf("Performing Sobel edge detection.\n");
      clock_gettime(CLOCK_REALTIME, &rtcStart);
   }

   int convergenceThreshold = SerialSobelEdgeDetection(inputImage, outputImageBlock, get_image_height(), get_image_width());

   if(myRank == masterProcessRank)
   {
      // Gather image pixels from all slaves

      clock_gettime(CLOCK_REALTIME, &rtcEnd);

      DisplayResults(
         communicatorSize,
         executionTime(rtcStart, rtcEnd),
         convergenceThreshold);

      // Write output image buffers. Closes files and frees buffers.
      FILE *outputFile = fopen(argv[2], "wb");
      write_bmp_file(outputFile, outputImageBlock);
   }
   else
   {
      // Send image pixels to master

      free(outputImageBlock)
   }


   MPI_Finalize();
}
