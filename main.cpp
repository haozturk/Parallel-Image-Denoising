/*
Student Name: Hasan Öztürk
Student Number: 2017400258
Compile Status: Compiling
Program Status: Working
Notes: I added the output of my program to the documentation. I used β = 0.8 and π = 0.15.
1000000 iterations for the yinyang image seems enough to me. For the lena image
20000 iterations seems better to me.
*/


#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>


using namespace std;

//This is the split function for parsing inputs
template <class Container>
void splitStr(const string& str, Container& cont)
{
    istringstream iss(str);
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(cont));
}

//Converts a matrix into a small matrix according to arguments and returns the sum
int cutAndSum(int array[200][200],int rowStart, int rowEnd, int colStart, int colEnd);
//Dynamically creates 2d array
int **alloc_2d_int(int rows, int cols);

int main(int argc, char* argv[]) {


    //ifstream infile("lena200_noisy.txt");
    //ifstream infile("matrix.txt");
    ifstream infile(argv[1]);

    srand(time(nullptr));
    vector<string> words;
    string line;
    int X[200][200];

    FILE *myfile;
    myfile = fopen(argv[2],"w");

    //MPI Initialization
    MPI_Init(NULL,NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = world_size - 1;

    double beta = atof(argv[3]);
    double pi = atof(argv[4]);
    double gamma = 0.5*log((1-pi)/pi);
    int T = 1000000;


    /*
     * In this if statement master slave takes the input and sends the input by dividing it into
     * number of slave processors. Then takes the denoised parts from the slaves and merges them.
     * Then prints the denosied image to the output file
     */
    if(world_rank==0){

        //Read Input to 2d Array X
        for(int i=0;i<200;i++){
            getline(infile,line);
            splitStr(line,words);
            for(int j=0;j<200;j++){
                X[i][j] = stoi(words[j]);
            }
            words.clear();
        }

        /*
         * In order to send the input matrix to the slaves, I dynamically allocated an array A and
         * copied the content of input matrix to A
         */
        int** A;
        A = alloc_2d_int(200,200);
        for(int i=0;i<200;i++){
            for(int j=0;j<200;j++){
                A[i][j] = X[i][j];
            }
        }

        //Send the input matrix to the slaves by dividing it into the number of processors
        for(int i=1;i <= N;i++){
            MPI_Send(&A[(200/N)*(i-1)][0],200*(200/N),MPI_INT,i,0,MPI_COMM_WORLD);
        }
        free(A);

        //// RECEIVE DENOISED IMAGES FROM SLAVES
        for(int t=1;t <= N;t++){
            int *subarray = new int[200*(200/N)];
            MPI_Recv(subarray,200*(200/N), MPI_INT, t, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //convert 1D array to 2D
            int D[200/N][200];
            for(int i=0;i<(200/N);i++){
                for(int j=0;j<200;j++){
                    D[i][j] = subarray[200*i + j];
                }
            }

            //Print the partial matrix to output file
            for(int i=0;i<200/N;i++){
                for(int j=0;j<200;j++){
                    fprintf(myfile,"%d ",D[i][j]);
                }
                fprintf(myfile,"\n");
            }
            delete [] subarray;
        }
    }
    /*
     * Slave processes execute this else block. They first take their part of the input matrix and make
     * operations on it and send it back to master
     */
    else{

        /*
         * Slaves take (200/N) * 200 matrix from the master an 1d array. Once they receive partial input,
         * They convert 1d array to 2d array
         */
        int *subarr = new int[200*(200/N)];

        MPI_Recv(subarr,200*(200/N), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /*
        printf("Process %d received elements: ", world_rank);
        for(int i = 0 ; i < 200*(200/N)  ; i++)
            printf("%d ", subarr[i]);
        printf("\n");*/


        //convert 1D array to 2D
        int X[200/N][200];
        for(int i=0;i<(200/N);i++){
            for(int j=0;j<200;j++){
                X[i][j] = subarr[200*i + j];
            }
        }


        //Z = X.copy()
        int Z[200/N][200];
        for(int i=0;i<200/N;i++){
            for(int j=0;j<200;j++){
                Z[i][j] = X[i][j];
            }
        }


        /*
         * Processors whose rank are 1 and N work at the boundaries of the image and we should handle
         * these processors seperately, because they just share 1 row with neighbors as opposed to internal processors
         * which share 2 rows.
         */

        if(world_rank == 1 || world_rank == N){

            if(world_rank ==1){

                /*
                 * Processor whose rank is 1 converts its matrix from (200/N) * 200 to (200/N) +1 * 200 by
                 * receiving 1 row from the below processor. Plus, it sends the boundary row to its neighbor.
                 * Before starting the iterations we first update our X matrix to extX which stands for
                 * extended X
                 */

                MPI_Send(&X[(200/N)-1][0],200,MPI_INT,world_rank+1,3,MPI_COMM_WORLD);

                int *bottomLineX = new int[200];
                MPI_Recv(bottomLineX,200, MPI_INT, world_rank+1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //Create the extended X array by taking 1 row from the neighbor
                //Extended X is (200/N + 1)*200
                int extX[(200/N)+1][200];
                //Take the bottomline
                for(int i=0;i<200;i++){
                    extX[200/N][i] = bottomLineX[i];
                }
                //Take the rest
                for(int i=0;i<(200/N);i++){
                    for(int j=0;j<200;j++){
                        extX[i][j] = X[i][j];
                    }
                }

                /*
                 * This is the main for loop for processor whose rank is 1. Iterations are achieved here and in each
                 * iteration neighbor pixels are shared.
                 */
                for(int t=0;t<T/N;t++){


                    MPI_Send(&Z[(200/N)-1][0],200,MPI_INT,world_rank+1,3,MPI_COMM_WORLD);

                    int *bottomLineZ = new int[200];
                    MPI_Recv(bottomLineZ,200, MPI_INT, world_rank+1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                    /*
                     * Calculations are with the extended Z matrix whose dimensions are (200/N) +1 * 200. In this part,
                     * we create extended Z.
                     */
                    int extZ[(200/N)+1][200];
                    //Take the topline
                    for(int i=0;i<200;i++){
                        extZ[200/N][i] = bottomLineZ[i];
                    }
                    //Take the rest
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            extZ[i][j] = Z[i][j];
                        }
                    }

                    //Pick a pixel randomly
                    int i,j;
                    i = rand() % 200/N+1;
                    j = rand() % 200;

                    //Main equation
                    double delta_E = -2*gamma*extX[i][j]*extZ[i][j] - 2*beta*extZ[i][j]*(cutAndSum(extZ,max(i-1,0),i+2,max(j-1,0),j+2)-extZ[i][j]);

                    //Flip the pixel with the probability
                    double r = ((double) rand() / (RAND_MAX));
                    if(log(r) < delta_E)
                        extZ[i][j] = -extZ[i][j];

                    //Convert the extended Z to original Z by deleting extra row
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            Z[i][j] = extZ[i][j];
                        }
                    }

                }

            }
            //if world rank == N
            else{

                /*
                 * Processor whose rank is N converts its matrix from (200/N) * 200 to (200/N) +1 * 200 by
                 * receiving 1 row from the above processor. Plus, it sends the boundary row to its neighbor.
                 * Before starting the iterations we first update our X matrix to extX which stands for
                 * extended X
                 */

                int *topLineX = new int[200];
                MPI_Recv(topLineX,200, MPI_INT, world_rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(&X[0][0],200,MPI_INT,world_rank-1,4,MPI_COMM_WORLD);

                //Create the extended X array by taking 1 row from the neighbor
                //Extended X is (200/N + 1)*200
                int extX[(200/N)+1][200];
                //Take the topline
                for(int i=0;i<200;i++){
                    extX[0][i] = topLineX[i];
                }
                //Take the rest
                for(int i=0;i<(200/N);i++){
                    for(int j=0;j<200;j++){
                        extX[i+1][j] = X[i][j];
                    }
                }

                /*
                 * This is the main for loop for processor whose rank is N. Iterations are achieved here and in each
                 * iteration neighbor pixels are shared.
                 */
                for(int t=0;t<T/N;t++){

                    int *topLineZ = new int[200];
                    MPI_Recv(topLineZ,200, MPI_INT, world_rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Send(&Z[0][0],200,MPI_INT,world_rank-1,4,MPI_COMM_WORLD);


                    /*
                     * Calculations are with the extended Z matrix whose dimensions are (200/N) +1 * 200. In this part,
                     * we create extended Z.
                     */
                    int extZ[(200/N)+1][200];
                    //Take the topline
                    for(int i=0;i<200;i++){
                        extZ[0][i] = topLineZ[i];
                    }
                    //Take the rest
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            extZ[i+1][j] = Z[i][j];
                        }
                    }

                    //Pick a pixel randomly
                    int i,j;
                    i = rand() % 200/N+1;
                    j = rand() % 200;

                    //Main equation
                    double delta_E = -2*gamma*extX[i][j]*extZ[i][j] - 2*beta*extZ[i][j]*(cutAndSum(extZ,max(i-1,0),i+2,max(j-1,0),j+2)-extZ[i][j]);

                    //Flip the pixel with the probability
                    double r = ((double) rand() / (RAND_MAX));
                    if(log(r) < delta_E)
                        extZ[i][j] = -extZ[i][j];

                    //Convert the extended Z to original Z by deleting extra row
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            Z[i][j] = extZ[i+1][j];
                        }
                    }

                }

            }

        }
        /* To avoid deadlock between processor, I divide the processors into to with the following manner:
         * Processors whose rank is odd does send - recv - send - recv and processors whose rank is even 
         * does recv - send - recv - send to communicate with each other. All odd ranked processors communicate
         * with even ranked processors and vice versa. Different from the boundary processors, these internal 
         * processors share two rows with two neighbors
        */
        else{
            //Even ranked processors
            if(world_rank % 2 ==0){

                //Receive the bottom row
                int *bottomLineX = new int[200];
                MPI_Recv(bottomLineX,200, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //Send the bottom row
                MPI_Send(&X[(200/N)-1][0],200,MPI_INT,world_rank+1,2,MPI_COMM_WORLD);

                //Receive the top row
                int *topLineX = new int[200];
                MPI_Recv(topLineX,200, MPI_INT, world_rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //Send the top row
                MPI_Send(&X[0][0],200,MPI_INT,world_rank-1,4,MPI_COMM_WORLD);

                //Create the extended X array by taking 2 rows from the neighbor
                //Extended X is (200/N + 2)*200
                int extX[(200/N)+2][200];
                //Take the topline
                for(int i=0;i<200;i++){
                    extX[0][i] = topLineX[i];
                }
                //Take the bottomline
                for(int i=0;i<200;i++){
                    extX[(200/N)+1][i] = bottomLineX[i];
                }
                //Copy the rest
                for(int i=0;i<(200/N);i++){
                    for(int j=0;j<200;j++){
                        extX[i+1][j] = X[i][j];
                    }
                }

                /*
                 * This is the main for loop for the internal processors whose rank are even. 
                 * Iterations are achieved here and in each iteration neighbor pixels are shared.
                 */
                for(int t=0;t<T/N;t++){

                    //Receive the bottomline
                    int *bottomLineZ = new int[200];
                    MPI_Recv(bottomLineZ,200, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    //Send the bottomline
                    MPI_Send(&Z[(200/N)-1][0],200,MPI_INT,world_rank+1,2,MPI_COMM_WORLD);

                    //Receive the topline
                    int *topLineZ = new int[200];
                    MPI_Recv(topLineZ,200, MPI_INT, world_rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    //Send the topline
                    MPI_Send(&Z[0][0],200,MPI_INT,world_rank-1,4,MPI_COMM_WORLD);


                    /*
                     * Calculations are with the extended Z matrix whose dimensions are (200/N) +2 * 200. In this part,
                     * we create extended Z.
                     */
                    int extZ[(200/N)+2][200];
                    //Take the topline
                    for(int i=0;i<200;i++){
                        extZ[0][i] = topLineZ[i];
                    }
                    //take the bottomline
                    for(int i=0;i<200;i++){
                        extZ[200/N+1][i] = bottomLineZ[i];
                    }
                    //Take the rest
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            extZ[i+1][j] = Z[i][j];
                        }
                    }

                    //Pick a random pixel
                    int i,j;
                    i = rand() % 200/N+1;
                    j = rand() % 200;

                    //Main equation
                    double delta_E = -2*gamma*extX[i][j]*extZ[i][j] - 2*beta*extZ[i][j]*(cutAndSum(extZ,max(i-1,0),i+2,max(j-1,0),j+2)-extZ[i][j]);

                    //Flip the pixel with a probability
                    double r = ((double) rand() / (RAND_MAX));
                    if(log(r) < delta_E)
                        extZ[i][j] = -extZ[i][j];

                    //Convert the extended Z to original Z by deleting extra row
                    for(int i=0;i<(200/N);i++) {
                        for (int j = 0; j < 200; j++) {
                            Z[i][j] = extZ[i + 1][j];
                        }
                    }

                }

            }
            //Processors whose rank is odd
            else{

                //Send the topline
                MPI_Send(&X[0][0],200,MPI_INT,world_rank-1,1,MPI_COMM_WORLD);

                //Receive the topline
                int *topLineX = new int[200];
                MPI_Recv(topLineX,200, MPI_INT, world_rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //Send the bottomline
                MPI_Send(&X[200/N-1][0],200,MPI_INT,world_rank+1,3,MPI_COMM_WORLD);

                //Receive the bottomline
                int *bottomLineX = new int[200];
                MPI_Recv(bottomLineX,200, MPI_INT, world_rank+1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //Create the extended X array by taking 2 rows from the neighbor
                //Extended X is (200/N + 2)*200
                int extX[(200/N)+2][200];
                for(int i=0;i<200;i++){
                    extX[0][i] = topLineX[i];
                }
                for(int i=0;i<200;i++){
                    extX[(200/N)+1][i] = bottomLineX[i];
                }
                for(int i=0;i<(200/N);i++){
                    for(int j=0;j<200;j++){
                        extX[i+1][j] = X[i][j];
                    }
                }

                /*
                 * This is the main for loop for the internal processors whose rank are even. 
                 * Iterations are achieved here and in each iteration neighbor pixels are shared.
                 */
                for(int t=0;t<T/N;t++){

                    //Send the topline
                    MPI_Send(&Z[0][0],200,MPI_INT,world_rank-1,1,MPI_COMM_WORLD);

                    //Receive the topline
                    int *topLineZ = new int[200];
                    MPI_Recv(topLineZ,200, MPI_INT, world_rank-1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    //Send the bottomline
                    MPI_Send(&Z[(200/N)-1][0],200,MPI_INT,world_rank+1,3,MPI_COMM_WORLD);

                    //Receive the bottomline
                    int *bottomLineZ = new int[200];
                    MPI_Recv(bottomLineZ,200, MPI_INT, world_rank+1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                    /*
                     * Calculations are with the extended Z matrix whose dimensions are (200/N) +2 * 200. In this part,
                     * we create extended Z.
                     */
                    int extZ[(200/N)+2][200];
                    //Take the topline
                    for(int i=0;i<200;i++){
                        extZ[0][i] = topLineZ[i];
                    }
                    //take the bottomline
                    for(int i=0;i<200;i++){
                        extZ[200/N+1][i] = bottomLineZ[i];
                    }
                    //Take the rest
                    for(int i=0;i<(200/N);i++){
                        for(int j=0;j<200;j++){
                            extZ[i+1][j] = Z[i][j];
                        }
                    }

                    //Pick a random pixel
                    int i,j;
                    i = rand() % 200/N+1;
                    j = rand() % 200;

                    //Main equation
                    double delta_E = -2*gamma*extX[i][j]*extZ[i][j] - 2*beta*extZ[i][j]*(cutAndSum(extZ,max(i-1,0),i+2,max(j-1,0),j+2)-extZ[i][j]);

                    //Flip the pixel with the probability
                    double r = ((double) rand() / (RAND_MAX));
                    if(log(r) < delta_E)
                        extZ[i][j] = -extZ[i][j];

                    //Convert the extended Z to original Z by deleting extra row
                    for(int i=0;i<(200/N);i++) {
                        for (int j = 0; j < 200; j++) {
                            Z[i][j] = extZ[i + 1][j];
                        }
                    }

                }
            }

        }

        //// SEND DENOISED IMAGES TO MASTER

        //In order to send the partial images I created dynamic array 
        int** D;
        D = alloc_2d_int(200/N,200);

        //Fill the dynamic array
        for(int i=0;i<200/N;i++){
            for(int j=0;j<200;j++){
                D[i][j] = Z[i][j];
            }
        }

        //send to master
        MPI_Send(&D[0][0],200*(200/N),MPI_INT,0,0,MPI_COMM_WORLD);

        free(D);

        delete [] subarr;
    }

    MPI_Finalize();
    fclose(myfile);

    return 0;
}

//Dynamically creates 2d array
int **alloc_2d_int(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

//Converts a matrix into a small matrix according to arguments and returns the sum
int cutAndSum(int array[200][200],int rowStart, int rowEnd, int colStart, int colEnd){

    int sum = 0;

    for(int i=rowStart;i < rowEnd;i++){
        for(int j=colStart;j<colEnd;j++){
            sum += array[i][j];
        }
    }
    return sum;
}
