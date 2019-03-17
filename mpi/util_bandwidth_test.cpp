/*!
* \brief Test bandwidth by point-to-point communications.
* \reference https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_bandwidth.c
*/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_TASKS      8192
/* Change the next four parameters to suit your case */
#define START_SIZE     100000
#define END_SIZE       1000000
#define INCREMENT     100000
#define ROUNDTRIPS    100

void GetPartner(int rank, int num_tasks, int *partner) {
  if (rank < num_tasks / 2)
    *partner = num_tasks / 2 + rank;
  if (rank >= num_tasks / 2)
    *partner = rank - num_tasks / 2;
}

void ShowExperimentInfo(int rank, int num_tasks, int partner, int start, int end, 
                        int incr, int rndtrps, int *task_pairs) {
  char host[MPI_MAX_PROCESSOR_NAME],
       host_map[MAX_TASKS][MPI_MAX_PROCESSOR_NAME];  
       
  int namelength;
  MPI_Get_processor_name(host, &namelength);
  MPI_Gather(&host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, &host_map,
    MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double resolution = MPI_Wtick();
    printf("\n******************** MPI Bandwidth Test ********************\n");
    printf("Message start size= %d bytes\n", start);
    printf("Message finish size= %d bytes\n", end);
    printf("Incremented by %d bytes per iteration\n", incr);
    printf("Roundtrips per iteration= %d\n", rndtrps);
    printf("MPI_Wtick resolution = %e\n", resolution);
    printf("************************************************************\n");
    for (int i = 0; i < num_tasks; i++)
      printf("task %4d is on %s partner=%4d\n", i, host_map[i], task_pairs[i]);
    printf("************************************************************\n");
  }
}

int main(int argc, char *argv[]) {
  
  /* Some initializations and error checking */  
  int num_tasks, rank, rc;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  if (num_tasks % 2 != 0) {
    printf("ERROR: Must be an even number of tasks!  Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(-1);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Set the basic parameters.
  int start = START_SIZE;
  int end = END_SIZE;
  int incr = INCREMENT;
  int rndtrps = ROUNDTRIPS;

  // Determine who my send/receive partner is and tell task 0.
  int partner;
  GetPartner(rank, num_tasks, &partner);
  int task_pairs[MAX_TASKS];  
  MPI_Gather(&partner, 1, MPI_INT, &task_pairs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  ShowExperimentInfo(rank, num_tasks, partner, start, end, incr, rndtrps, task_pairs);

  // Start working...
  MPI_Status status;
  int tag = 1;
  char msgbuf[END_SIZE]; 
  for (int i = 0; i < end; i++)
    msgbuf[i] = 'x';

  /*************************** first half of tasks *****************************/
  /* These tasks send/receive messages with their partner task, and then do a  */
  /* few bandwidth calculations based upon message size and timings.           */
  if (rank < num_tasks / 2) {   
    double timings[MAX_TASKS / 2][3];
    double bestbw, worstbw, totalbw, avgbw;
    double bestall, avgall, worstall;

    for (int n = start; n <= end; n = n + incr) {
      bestbw = totalbw = avgbw = 0.0;
      worstbw = .99E+99;
      int nbytes = sizeof(char) * n;
      for (int i = 1; i <= rndtrps; i++) {
        double t = MPI_Wtime();
        MPI_Send(&msgbuf, n, MPI_CHAR, partner, tag, MPI_COMM_WORLD);
        MPI_Recv(&msgbuf, n, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status);
        double bw = ((double)nbytes * 2) / (MPI_Wtime() - t);
        totalbw = totalbw + bw;
        if (bw > bestbw) bestbw = bw;
        if (bw < worstbw) worstbw = bw;
      }
      /* Convert to megabytes per second */
      bestbw = bestbw / 1000000.0;
      avgbw = (totalbw / 1000000.0) / (double)rndtrps;
      worstbw = worstbw / 1000000.0;

      /* Task 0 collects timings from all relevant tasks */
      if (rank == 0) {
        /* Keep track of my own timings first */
        timings[0][0] = bestbw;
        timings[0][1] = avgbw;
        timings[0][2] = worstbw;
        /* Initialize overall averages */
        bestall = avgall = worstall = 0.0;
        /* Now receive timings from other tasks and print results. Note */
        /* that this loop will be appropriately skipped if there are    */
        /* only two tasks. */
        for (int j = 1; j < num_tasks / 2; j++)
          MPI_Recv(&timings[j], 3, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
        printf("***Message size: %8d *** best  /  avg  / worst (MB/sec)\n", n);
        for (int j = 0; j < num_tasks / 2; j++) {
          printf("   task pair: %4d - %4d:    %4.2f / %4.2f / %4.2f \n",
            j, task_pairs[j], timings[j][0], timings[j][1], timings[j][2]);
          bestall += timings[j][0];
          avgall += timings[j][1];
          worstall += timings[j][2];
        }
        printf("   OVERALL AVERAGES:          %4.2f / %4.2f / %4.2f \n\n",
          bestall / (num_tasks / 2), avgall / (num_tasks / 2), worstall / (num_tasks / 2));
      }
      else {
        /* Other tasks send their timings to task 0 */
        double tmptimes[3];
        tmptimes[0] = bestbw;
        tmptimes[1] = avgbw;
        tmptimes[2] = worstbw;
        MPI_Send(tmptimes, 3, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
      }
    }
  }

  /**************************** second half of tasks ***************************/
  /* These tasks do nothing more than send and receive with their partner task */
  if (rank >= num_tasks / 2) {
    for (int n = start; n <= end; n = n + incr) {
      for (int i = 1; i <= rndtrps; i++) {
        MPI_Recv(&msgbuf, n, MPI_CHAR, partner, tag, MPI_COMM_WORLD, &status);
        MPI_Send(&msgbuf, n, MPI_CHAR, partner, tag, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Finalize();

}  /* end of main */
