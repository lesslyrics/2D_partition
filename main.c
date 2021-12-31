#include <stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define SIZE_1 4            // SIZE_1 should be a multiple of number of nodes
#define SIZE_2 1            // SIZE_2 is always 1 as we consider vector

#define TURN_ON_DEBUG 0

#define MASTER 1
#define SLAVE 2

MPI_Status status;

static double a[SIZE_1][SIZE_1];
static double b[SIZE_1][SIZE_2];
static double c[SIZE_1][SIZE_2];
static double b_transpose[SIZE_2][SIZE_1];

static void init_matrices() {
    int i, j;
    for (i = 0; i < SIZE_1; i++) {
        for (j = 0; j < SIZE_1; j++) {
            a[i][j] = 1.0;
            if (i >= SIZE_1 / 2) a[i][j] = 2.0;

        }

        j = 0;
        b[i][j] = 1.0;
        c[i][j] = 0.0;
        b_transpose[j][i] = b[i][j]; // it is easier to send rows than columns
    }


    printf("\nInitial matrix A:\n");
    for (i = 0; i < SIZE_1; i++) {
        for (j = 0; j < SIZE_1; j++) {
            printf("%.2f ", a[i][j]);
        }
        printf("\n");
    }

    printf("\nInitial vector B:\n");
    for (i = 0; i < SIZE_1; i++) {
        j = 0;
        printf("%.2f\n", b[i][j]);
    }

}


static void print_result() {
    int i, j;
    printf("\n Final vector is:\n");
    for (i = 0; i < SIZE_1; i++) {
        j = 0;
        printf("%7.2f\n", c[i][j]);
    }
}

int main(int argc, char **argv) {
    int p_rank, nodes_num;
    int message_type;
    int cols;    // columns per worker
    int rows;   // rows per worker
    int dest, src;
    int offset_1, offset_2, offset_1_updated;
    double start_t, end_t;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    if (nodes_num == 1) {
        init_matrices();
        printf("\n SIZE_1 = %d, number of  nodes = %d\n", SIZE_1, nodes_num);
        start_t = MPI_Wtime();

        for (i = 0; i < SIZE_1; i++) {
            j = 0;
            c[i][j] = 0.0;
            for (k = 0; k < SIZE_1; k++) {
                printf("%f", c[i][j]);
                printf("%f", a[i][k]);
                printf("%f", b[k][j]);

                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
            printf("%f", c[i][j]);
        }
        end_t = MPI_Wtime();

        if (TURN_ON_DEBUG)
            print_result();

        printf("\n Execution time on %d nodes is %f\n", nodes_num, end_t - start_t);
    }

    if (nodes_num == 2) {
        if (p_rank == 0) {
            init_matrices();
            printf("\n SIZE_1 = %d, number of  nodes = %d\n", SIZE_1, nodes_num);
            start_t = MPI_Wtime();
            message_type = MASTER;

            rows = SIZE_1;
            cols = SIZE_1;

            offset_1 = rows;
            offset_2 = cols;
            offset_1_updated = 0;

            for (dest = 1; dest < nodes_num; dest++) {
                MPI_Send(&offset_1_updated, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&offset_2, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&a[0][0], SIZE_1 * SIZE_1, MPI_DOUBLE, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&b_transpose[offset_2][0], offset_2 * SIZE_1, MPI_DOUBLE, dest, message_type, MPI_COMM_WORLD);
            }
            // calculate on master
            for (i = 0; i < rows; i++) {
                j = 0;
                for (k = 0; k < SIZE_1; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
                }
            }

            // get result from slave
            message_type = SLAVE;
            for (src = 1; src < nodes_num; src++) {
                MPI_Recv(&offset_1_updated, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&offset_2, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                for (i = offset_1_updated; i < rows; i++) {
                    MPI_Recv(&c[i][offset_2], cols, MPI_DOUBLE, src, message_type, MPI_COMM_WORLD, &status);

                }

            }

            end_t = MPI_Wtime();

            if (TURN_ON_DEBUG)
                print_result();

            printf("\n Execution time on %d nodes is %f\n", nodes_num, end_t - start_t);
        }
        if (p_rank > 0) {
            message_type = MASTER;
            MPI_Recv(&offset_1_updated, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&offset_2, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&a[0][0], SIZE_1 * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&b_transpose[offset_2][0], offset_2 * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD,
                     &status);

            // calculate on slaves
            for (i = offset_1_updated; i < rows; i++) {
                for (j = offset_2; j < SIZE_2; j++) {
                    for (k = 0; k < SIZE_1; k++) {
                        c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
                    }
                }
            }

            if (TURN_ON_DEBUG) {
                printf("\n Rank is %d, calculated matrix part is:\n", p_rank);
                for (i = offset_1_updated; i < rows; i++)
                    for (j = offset_2; j < SIZE_2; j++)
                        printf("%7.2f", c[i][j]);
                printf("\n");
            }

            message_type = SLAVE;
            MPI_Send(&offset_1_updated, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&offset_2, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            for (i = offset_1_updated; i < rows; i++) {
                MPI_Send(&c[i][offset_2], offset_2, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD);
            }
        }

    }

    if (nodes_num >= 4) {
        if (p_rank == 0)         // job on master
        {
            init_matrices();
            printf("\nSIZE_1 = %d, number of nodes = %d\n", SIZE_1, nodes_num);

            start_t = MPI_Wtime();

            message_type = MASTER;
            rows = SIZE_1;        // Number of rows in matrix A for each node
            cols = SIZE_1;       // Number of columns in matrix B (number of rows in B-Transpose) for each node

            offset_1 = rows;
            offset_2 = cols;

            offset_1_updated = 0;

            for (dest = 1; dest < nodes_num / 2; dest++) { // Send data to first subset of slaves
                MPI_Send(&offset_1, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&cols, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&a[offset_1][0], rows * SIZE_1, MPI_DOUBLE, dest, message_type,
                         MPI_COMM_WORLD);      // Starting point in A
                MPI_Send(&b_transpose[0][0], cols * SIZE_1, MPI_DOUBLE, dest, message_type,
                         MPI_COMM_WORLD); // num of columns in vector B is equal to number of rows in B transposed
                offset_1 = offset_1 + rows;
            }

            for (dest = nodes_num / 2; dest < nodes_num; dest++) { // send data to the second subset of slaves
                MPI_Send(&offset_1_updated, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&offset_2, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&cols, 1, MPI_INT, dest, message_type, MPI_COMM_WORLD);
                MPI_Send(&a[offset_1_updated][0], rows * SIZE_1, MPI_DOUBLE, dest, message_type,
                         MPI_COMM_WORLD); // Starting point in A
                MPI_Send(&b_transpose[offset_2][0], cols * SIZE_1, MPI_DOUBLE, dest, message_type,
                         MPI_COMM_WORLD);
                offset_1_updated = offset_1_updated + rows;
            }

            // calculation on master
            for (i = 0; i < rows; i++) {
                j = 0;
                for (k = 0; k < SIZE_1; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
                }

            }


            // get part of the results from slaves
            message_type = SLAVE;
            for (src = 1; src < nodes_num / 2; src++) {
                MPI_Recv(&offset_1, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&cols, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                for (i = offset_1; i < offset_1 + rows; i++) {
                    MPI_Recv(&c[i][0], cols, MPI_DOUBLE, src, message_type, MPI_COMM_WORLD, &status);
                }

                if (TURN_ON_DEBUG) {
                    printf("\n Received this part from process %d \n", src);
                    for (i = offset_1; i < offset_1 + rows; i++)
                        for (j = 0; j < cols; j++)
                            printf("%7.2f", c[i][j]);
                    printf("\n");
                }
            }
            if (TURN_ON_DEBUG) {
                printf("\n Rank 0 calculated\n");
                for (i = 0; i < rows; i++)
                    for (j = 0; j < cols; j++)
                        printf("%7.2f", c[i][j]);
                printf("\n");
            }
            message_type = SLAVE;

            for (src = nodes_num / 2; src < nodes_num; src++) {
                MPI_Recv(&offset_1_updated, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&rows, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&offset_2, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                MPI_Recv(&cols, 1, MPI_INT, src, message_type, MPI_COMM_WORLD, &status);
                for (i = offset_1_updated; i < offset_1_updated + rows; i++) {
                    MPI_Recv(&c[i][offset_2], cols, MPI_DOUBLE, src, message_type, MPI_COMM_WORLD, &status);
                }
                if (TURN_ON_DEBUG) {
                    printf("\n Received this part from process %d \n", src);
                    for (i = offset_1_updated; i < rows; i++)
                        for (j = offset_2; j < SIZE_2; j++)
                            printf("%7.2f", c[i][j]);
                    printf("\n");
                }
            }

            if (TURN_ON_DEBUG) {
                printf("\n Rank 0 calculated\n");
                for (i = 0; i < rows; i++)
                    for (j = 0; j < cols; j++)
                        printf("%7.2f", c[i][j]);
                printf("\n");
            }

            end_t = MPI_Wtime();

            if (TURN_ON_DEBUG)
                print_result();

            printf("\n Execution time on %d nodes is: %f ", nodes_num, end_t - start_t);
        } else if (p_rank > 0 && p_rank < nodes_num / 2) // first subset of slaves
        {
            // get data from master
            message_type = MASTER;

            rows = SIZE_1;
            cols = SIZE_2;

            offset_1 = rows;
            offset_2 = cols;
            offset_1_updated = 0;

            message_type = MASTER;
            MPI_Recv(&offset_1, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&a[offset_1][0], rows * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&b_transpose[0][0], cols * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD, &status);

            // slaves calculation

            for (i = offset_1; i < offset_1 + rows; i++) {
                for (j = 0; j < cols; j++) {
                    for (k = 0; k < SIZE_1; k++) {
                        c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
                    }
                }
            }

            if (TURN_ON_DEBUG) {
                printf("\n Rank is %d Calculated matrix part is:\n", p_rank);
                for (i = offset_1; i < offset_1 + rows; i++)
                    for (j = 0; j < cols; j++)
                        printf("%7.2f", c[i][j]);
                printf("\n");
            }
            // sending slaves part of the results back to the master

            message_type = SLAVE;
            MPI_Send(&offset_1, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            for (i = offset_1; i < offset_1 + rows; i++) {
                MPI_Send(&c[i][0], cols, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD);
            }
        } else if (p_rank >= nodes_num / 2) // second subset of slaves
        {
            // get data from master
            message_type = MASTER;
            MPI_Recv(&offset_1_updated, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&offset_2, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&a[offset_1_updated][0], rows * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD, &status);
            MPI_Recv(&b_transpose[offset_2][0], cols * SIZE_1, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD, &status);

            // slaves calculations part

            for (i = offset_1_updated; i < offset_1_updated + rows; i++) {
                for (j = offset_2; j < SIZE_2; j++) {
                    for (k = 0; k < SIZE_1; k++) {
                        c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
                    }
                }
            }

            if (TURN_ON_DEBUG) {
                printf("\n Rank is %d Calculated matrix part is:\n", p_rank);
                for (i = offset_1_updated; i < rows; i++)
                    for (j = offset_2; j < SIZE_2; j++)
                        printf("%7.2f", c[i][j]);
                printf("\n");
            }


            // send results back to the master
            message_type = SLAVE;
            MPI_Send(&offset_1_updated, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&offset_2, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, 0, message_type, MPI_COMM_WORLD);
            for (i = offset_1_updated; i < offset_1_updated + rows; i++) {
                MPI_Send(&c[i][offset_2], cols, MPI_DOUBLE, 0, message_type, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;

}