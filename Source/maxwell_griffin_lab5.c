/***
 * File: maxwell_griffin_lab4p2.cu
 * Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
 *       serial algorithm, and once by a massively parallel CUDA algorithm.
 */

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

static struct timespec rtcSerialStart;
static struct timespec rtcSerialEnd;
static struct timespec rtcParallelStart;
static struct timespec rtcParallelEnd;

__device__ int Sobel_Gx[3][3] = {
   { -1, 0, 1 },
   { -2, 0, 2 },
   { -1, 0, 1 }
};

__device__ int Sobel_Gy[3][3] = {
   {  1,  2,  1 },
   {  0,  0,  0 },
   { -1, -2, -1 }
};

/*
 * Display all header information and matrix and CUDA parameters.
 *
 * @param inputFile -- name of the input image
 * @param serialOutputFile -- name of the serial output image
 * @param parallelOutputFile -- name of the parallel output image
 * @param imageHeight -- in pixels
 * @param imageWidth -- in pixels
 */
void DisplayParameters(
   char *inputFile,
   char *serialOutputFile,
   char *cudaOutputFile,
   int imageHeight,
   int imageWidth)
{
   printf("********************************************************************************\n");
   printf("lab4p2: serial vs. CUDA Sobel edge detection.\n");
   printf("\n");
   printf("Input image: %s \t(Height: %d pixels, width: %d pixels)\n", inputFile, imageHeight, imageWidth);
   printf("Serial output image: \t%s\n", serialOutputFile);
   printf("CUDA output image: \t%s\n", cudaOutputFile);
   printf("\n");
   printf("CUDA compute structure:\n");
   printf("|-- with %d grid\n", CUDA_GRIDS);
   printf("    |-- with %d blocks\n", CUDA_BLOCKS_PER_GRID);
   printf("        |-- with %d threads per block\n", CUDA_THREADS_PER_BLOCK);
   printf("\n");
}

/*
 * Display the timing and convergence results to the screen.
 *
 * @param serialConvergenceThreshold
 * @param serialConvergenceThreshold
 */
void DisplayResults(
   int serialConvergenceThreshold,
   int parallelConvergenceThreshold)
{
   printf("Time taken for serial Sobel edge detection: %lf\n",
      (LINEARIZE(rtcSerialEnd.tv_sec, rtcSerialEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcSerialStart.tv_sec, rtcSerialStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));

   printf("Convergence Threshold: %d\n", serialConvergenceThreshold);
   printf("\n");

   printf("Time taken for CUDA Sobel edge detection: %lf\n",
      (LINEARIZE(rtcParallelEnd.tv_sec, rtcParallelEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcParallelStart.tv_sec, rtcParallelStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));

   printf("Convergence Threshold: %d\n", parallelConvergenceThreshold);
   printf("********************************************************************************\n");
}

/*
 * Serial algorithm to keep perform a Sobel edge detection on an input pixel
 * buffer at different brightness thresholds until a certain percentage of
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
 * Massively parallel CUDA kernel function that performs a Sobel edge detection
 * on a group of pixels.
 *
 * @param input -- device input pixel buffer
 * @param output -- device output pixel buffer
 * @param height -- height of pixel image
 * @param width -- width of pixel image
 * @param gradientThreshold -- threshold to turn pixels black or white
 */
__global__ void CudaSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width, int gradientThreshold)
{
   int row = 0;
   for(int i = 0; row < (height - 1); i++)
   {
      // Let the blockIdx increment beyond its dimension for cyclic distribution of the test pixels
      int blockRow = (i * gridDim.x) + blockIdx.x;

      // Calculate the row/col in the image buffer that this thread's stencil's center is on
      row = (LINEARIZE(blockRow, threadIdx.x, blockDim.x) / (width - 2)) + 1;
      int col = (LINEARIZE(blockRow, threadIdx.x, blockDim.x) % (width - 2)) + 1;

      // Calculate Sobel magnitude of gradient directly, instead of using Sobel_Magnitude utility
      double Gx = (Sobel_Gx[0][0] * input[LINEARIZE(row - 1, col - 1, width)])
         + (Sobel_Gx[0][2] * input[LINEARIZE(row - 1, col + 1, width)])
         + (Sobel_Gx[1][0] * input[LINEARIZE(row, col - 1, width)])
         + (Sobel_Gx[1][2] * input[LINEARIZE(row, col + 1, width)])
         + (Sobel_Gx[2][0] * input[LINEARIZE(row + 1, col - 1, width)])
         + (Sobel_Gx[2][2] * input[LINEARIZE(row + 1, col + 1, width)]);

      double Gy = (Sobel_Gy[0][0] * input[LINEARIZE(row - 1, col - 1, width)])
         + (Sobel_Gy[0][1] * input[LINEARIZE(row - 1, col, width)])
         + (Sobel_Gy[0][2] * input[LINEARIZE(row - 1, col + 1, width)])
         + (Sobel_Gy[2][0] * input[LINEARIZE(row + 1, col - 1, width)])
         + (Sobel_Gy[2][1] * input[LINEARIZE(row + 1, col, width)])
         + (Sobel_Gy[2][2] * input[LINEARIZE(row + 1, col + 1, width)]);

      if(((Gx * Gx) + (Gy * Gy)) > (gradientThreshold * gradientThreshold))
      {
         output[LINEARIZE(row, col, width)] = PIXEL_WHITE;
      }
      else
      {
         output[LINEARIZE(row, col, width)] = PIXEL_BLACK;
      }
   }
}

/*
 * Parallel algorithm to keep perform a Sobel edge detection on an input pixel
 * buffer at different brightness thresholds until a certain percentage of
 * pixels in the output pixel buffer are black.
 *
 * @param input -- input pixel buffer
 * @param output -- output pixel buffer
 * @param height -- height of pixel image
 * @param width -- width of pixel image
 * @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD pixels are black
 */
__host__ int ParallelSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width)
{
   int numBlocks = CUDA_BLOCKS_PER_GRID;  // = ?
   int threadsPerBlock = CUDA_THREADS_PER_BLOCK;  // = ?
   size_t imageMemSize =  height * width * sizeof(uint8_t);
   uint8_t *deviceInputImage, *deviceOutputImage;

   // Allocate device memory
   cudaMalloc((void **)&deviceInputImage, imageMemSize);
   cudaMalloc((void **)&deviceOutputImage, imageMemSize);

   // Copy host input array to device
   cudaMemcpy(deviceInputImage, input, imageMemSize, cudaMemcpyHostToDevice);

   dim3 dimGrid(numBlocks);
   dim3 dimBlock(threadsPerBlock);

   int gradientThreshold, blackPixelCount = 0;
   for(gradientThreshold = 0; blackPixelCount < (height * width * 3 / 4); gradientThreshold++)
   {
      // Launch Kernel
      CudaSobelEdgeDetection<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, height, width, gradientThreshold);

      // Copy device results array back to host
      cudaMemcpy(output, deviceOutputImage, imageMemSize, cudaMemcpyDeviceToHost);

      // Count number of black pixels
      blackPixelCount = 0;
      for(int row = 1; row < (height - 1); row++)
      {
         for(int col = 1; col < (width - 1); col++)
         {
            if(output[LINEARIZE(row, col, width)] == PIXEL_BLACK)
            {
               blackPixelCount++;
            }
         }
      }
   }

   return gradientThreshold;
}

/*
* Main function.
*/
int main(int argc, char* argv[])
{
   // Check for correct number of comand line args
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
}
