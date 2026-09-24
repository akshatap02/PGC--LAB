# Matrix Multiplication — Code and Explanations (Parts A–C)

All three programs compute `C = A × B` for 4000×4000 matrices where every element of `A` and `B` is `1.0`, so every element of `C` should come out to `4000.00`. That's the whole point of using `1.0` everywhere — it gives you a free correctness check without needing to reason about real numbers. This section walks through the actual code for each version and what it's doing differently under the hood.

---

## Part A — Sequential (`matrix_sequential.c`)
One core, no parallelism, nothing shared or split up — just the raw triple loop running start to finish. This exists purely to give the other versions something to be measured against.

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    clock_t start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }

    start = clock();

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

    end = clock();

    printf("Execution Time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
```

**What it does:**
- `A`, `B`, and `C` are stored as flat 1D arrays (`A[i*N+j]`) instead of real 2D arrays, and allocated with `malloc`. This isn't just style — a 4000×4000 2D array declared normally would try to sit on the stack and blow it up immediately. Heap allocation sidesteps that.
- The triple-nested loop is just matrix multiplication written out literally: `C[i][j] = Σₖ A[i][k] · B[k][j]`.
- `clock()` is measuring CPU time here, which is fine — there's only one thread doing anything, so CPU time and wall-clock time are basically the same.
- This is O(N³) work. At N = 4000 that's around 64 billion multiply-add operations, done strictly one after another. That number alone tells you why this version is slow.

---

## Part B — OpenMP (`matrix_openmp.c`)
Same machine, same process, but now several threads split up the rows and work on them at once. This works because threads inside one process share memory — a thread doesn't need its own copy of `A` or `B`, it can just read them where they already sit. The only overhead is threads briefly syncing up when they're done, which is why OpenMP tends to scale close to linearly (8 threads getting you close to 8x).

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    double start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }

    start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

    end = omp_get_wtime();

    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %f seconds\n", end - start);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
```

**What it does:**
- The math didn't change at all from Part A — the only new line is `#pragma omp parallel for`, which tells OpenMP to split the outer `i` loop across threads. Each thread just gets handed a block of rows to work through.
- `private(j, k)` matters because without it, threads would be fighting over the same `j` and `k` counters. `A`, `B`, and `C` stay shared on purpose — that's safe here because no two threads ever write to the same row of `C`.
- How many threads you get is set by the `OMP_NUM_THREADS` environment variable, not by anything in the code. The reference run used 8, so bumping it up or down doesn't require recompiling.
- `omp_get_wtime()` is used instead of `clock()` because now that multiple threads are running at once, CPU time stops meaning what you'd expect — wall-clock time is what actually matters.
- There's zero data copying anywhere in this version. Every thread already sees the same `A` and `B` in memory, so the only coordination is a quick join at the end.

---

## Part C — MPI (`matrix_mpi.c`)
Now instead of threads inside one process, there are 4 separate processes, and they could just as well be running on 4 different machines. Separate processes means separate memory — process 2 can't just peek at what process 0 has in RAM, so anything it needs has to be explicitly sent over. Rank 0 splits `A`'s rows into chunks and sends one to each rank, broadcasts a full copy of `B` to everyone (since every rank needs all of it to compute its rows), each rank multiplies its own chunk on its own, and then the results all get sent back to rank 0 to be reassembled. The compute itself is exactly as parallel as OpenMP's — the difference is all that sending and receiving is pure overhead OpenMP never has to pay, which is why MPI can end up slower than OpenMP even with the same number of workers, unless the problem is big enough that the math outweighs the communication.

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, size, i, j, k, rows_per_process;
    char hostname[256];
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, sizeof(hostname));

    rows_per_process = N / size;

    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    local_C = (double *)malloc(rows_per_process * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));

    if (rank == 0)
    {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE,
                local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Rank %d on %s computing %d rows\n", rank, hostname, rows_per_process);

    for (i = 0; i < rows_per_process; i++)
        for (j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0.0;
            for (k = 0; k < N; k++)
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
        }

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE,
               C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Execution Time = %f seconds\n", end - start);
        printf("Verification C[0][0] = %.2f\n", C[0]);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
```

**What it does:**
- Each of the 4 ranks is its own process with its own memory space. Nothing is shared automatically the way it is with threads — every rank has to explicitly ask for the data it needs.
- Only rank 0 bothers allocating the full-size `A` and `C`. Every other rank only ever holds its own small slice — they never see the whole matrix.
- `MPI_Scatter` is what splits `A` into 4 pieces of 1000 rows each and sends one to each rank (rank 0 keeps a piece too, it doesn't get special treatment there).
- `MPI_Bcast` sends the whole `B` matrix to every single rank. This has to be the full matrix, not a slice, because computing even one row of `C` needs every column of `B`.
- Once the data's in place, the multiply itself needs no communication at all — each rank just grinds through its own `local_C` on its own. This is the "embarrassingly parallel" part.
- `MPI_Gather` is the mirror image of the scatter — it pulls everyone's `local_C` back to rank 0 and reassembles it into the full result.
- The `MPI_Barrier` calls bookending the timed region, combined with `MPI_Wtime()`, make sure the clock doesn't start until every rank is actually ready, and doesn't stop until every rank has reported in. So the number you get reflects the slowest rank, not just whatever rank 0 happened to see.
- All of that scattering, broadcasting, and gathering is exactly why this version can lose to OpenMP even with the same number of workers — shipping a 128 MB copy of `B` to every rank and shuffling `A` and `C` back and forth costs real time that OpenMP's threads simply never spend, since they're all looking at the same memory already.

---
