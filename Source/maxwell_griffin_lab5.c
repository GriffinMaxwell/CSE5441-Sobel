/***
 * File: maxwell_griffin_lab5.c
 * Desc: Performs a Sobel edge detection operation on a .bmp using MPI for
 *       parallelization.
 */

#include <mpi.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "read_bmp.h"
#include "Sobel.h"

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
   return (LINEARIZE(end.tv_sec, end.tv_nsec, NS_PER_SEC)
      - LINEARIZE(start.tv_sec, start.tv_nsec, NS_PER_SEC))
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
 * Determines if a pixel is on the border of an image.
 *
 * @param offset -- offset of the pixel within the image buffer
 * @param `height -- `total height of the image
 * @param width -- total width of the image
 */
bool IsBorderPixel(int offset, int height, int width)
{
   return ((offset < width)
      || (offset >= (width * (height - 1)))
      || (offset % width == 0)
      || (offset % width == (width - 1)));
}

/*
 * Using MPI parallelization, perform Sobel edge detections on a block of pixels
 * in an input pixel buffer at increasing brightness thresholds until
 * convergence, i.e. 75% of pixels in the output pixel buffer are black.
 *
 * @param input -- input pixel buffer
 * @param output -- output pixel buffer
 * @param height -- height of pixel image
 * @param width -- width of pixel image
 * @param initialOffset -- the offset within the input buffer to start testing pixels
 * @param blockSize -- i.e. the number of pixels in the output pixel buffer
 * @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD pixels are black
 */
static int BlockSobelEdgeDetection(
   uint8_t *input,
   uint8_t *output,
   int height,
   int width,
   int initialOffset,
   int blockSize)
{
   int gradientThreshold, myBlackPixelCount = 0, totalBlackPixelCount = 0;
   for(gradientThreshold = 0; totalBlackPixelCount < (height * width * 3 / 4); gradientThreshold++)
   {
      myBlackPixelCount = 0;

      // Initialize stencil to sit centered at input[1][1]
      Stencil_t pixel = {
         .top =    input + initialOffset - width,
         .middle = input + initialOffset,
         .bottom = input + initialOffset + width
      };

      int i;
      for(i = 0; i < blockSize; i++)
      {
         if(IsBorderPixel(i, height, width))
         {
            output[i] = PIXEL_BLACK;
         }
         else
         {
            if(Sobel_Magnitude(&pixel) > gradientThreshold)
            {
               output[i] = PIXEL_WHITE;
            }
            else
            {
               output[i] = PIXEL_BLACK;
               myBlackPixelCount++;
            }
         }

         Stencil_MoveRight(&pixel);
      }

      // Get the total black pixel count from all processes
      MPI_Allreduce(&myBlackPixelCount, &totalBlackPixelCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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

   struct timespec rtcStart, rtcEnd;
   int communicatorSize, myRank, masterProcessRank;
   int slaveBlockSize, myBlockSize, myBufferSize;
   int myInputOffset, myOutputOffset, numExtraPixels;
   int convergenceThreshold;
   uint8_t *inputImageBuffer, *outputImageBuffer;
   FILE *inputFile;

   MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   // Define the "master process" as the process with the largest rank
   // Rank is zero indexed, so largest rank is communicator size - 1
   masterProcessRank = communicatorSize - 1;

   // Check for correct number of remaining command line args
   if (argc != 3)
   {
      printf("Error: Incorrect arguments: <input.bmp> <output.bmp>\n");
      return 0;
   }

   // Open the input file specified by the command line arg
   inputFile = fopen(argv[1], "rb");
   if(inputFile == NULL)
   {
      printf("Error: %s could not be opened for reading.", argv[1]);
   }

   // Read input image, determine block size, and allocate output image buffer
   inputImageBuffer = (uint8_t *)read_bmp_file(inputFile);

   // Divvy out blocks of pixels to each process.
   slaveBlockSize = get_num_pixel() / communicatorSize;
   numExtraPixels = get_num_pixel() % communicatorSize;

   // Calculate process-specific variables
   // The master process gets the whole image buffer, plus the extra pixels from
   // uneven division between the blocks
   myBlockSize = (myRank == masterProcessRank) ? slaveBlockSize + numExtraPixels : slaveBlockSize;
   myBufferSize = (myRank == masterProcessRank) ? get_num_pixel() : myBlockSize;
   myInputOffset = myRank * slaveBlockSize;
   myOutputOffset = (myRank == masterProcessRank) ? myInputOffset : 0;

   outputImageBuffer = (uint8_t *)malloc(myBufferSize);

   // Wait for all processes to read the image and allocate buffers
   MPI_Barrier(MPI_COMM_WORLD);

   if(myRank == masterProcessRank)
   {
      DisplayParameters(argv[1], argv[2], get_image_height(), get_image_width());

      printf("Performing Sobel edge detection.\n");
      clock_gettime(CLOCK_REALTIME, &rtcStart);
   }

   convergenceThreshold = BlockSobelEdgeDetection(
      inputImageBuffer,
      &outputImageBuffer[myOutputOffset],
      get_image_height(),
      get_image_width(),
      myInputOffset,
      myBlockSize);

   // Fill output pixel array with the pixels from all the processes
   // Note: the the master process' pixels were written to their final spot in
   // this buffer so they won't be stomped on and the extra pixels are already at the very end of the buffer
   MPI_Gather(
      &outputImageBuffer[myOutputOffset], slaveBlockSize, MPI_UNSIGNED_CHAR,
      outputImageBuffer, slaveBlockSize, MPI_UNSIGNED_CHAR,
      masterProcessRank, MPI_COMM_WORLD);

   if(myRank == masterProcessRank)
   {
      clock_gettime(CLOCK_REALTIME, &rtcEnd);

      DisplayResults(
         communicatorSize,
         CalculateExecutionTime(rtcStart, rtcEnd),
         convergenceThreshold);

      // Write output image buffers. Closes files and frees buffers.
      FILE *outputFile = fopen(argv[2], "wb");
      write_bmp_file(outputFile, outputImageBuffer);
   }
   else
   {
      free(outputImageBuffer);   // outputImageBuffer is already freed in master process by write_bmp_file()
   }

   MPI_Finalize();
}
