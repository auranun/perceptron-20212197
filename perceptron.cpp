#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUT_NODE 2
#define MAXNUM_NUM_DATA_SET 4

#define LEARNING_RATE 0.01
#define MAX_EPOCH 10000

void initw(double weight[NUM_INPUT_NODE + 1]);
int getdata(double in[][NUM_INPUT_NODE + 1], double desired[]);
double forward(double x[NUM_INPUT_NODE + 1],
               double w[NUM_INPUT_NODE + 1]);
double activationfunc(double sum);

void neural_forward(double in[][NUM_INPUT_NODE + 1],
                    double w[NUM_INPUT_NODE + 1], int n_of_set,
                    double out[MAXNUM_NUM_DATA_SET]);
double neural_learning(double in[][NUM_INPUT_NODE + 1],
                       double w[NUM_INPUT_NODE + 1], int n_of_set,
                       double dout[MAXNUM_NUM_DATA_SET]);

int main()
{
    double x[MAXNUM_NUM_DATA_SET][NUM_INPUT_NODE + 1];
    double w[NUM_INPUT_NODE + 1];
    double desired[MAXNUM_NUM_DATA_SET];
    double output[MAXNUM_NUM_DATA_SET];

    double O;
    double err;

    int i, j;
    int n_of_set;

    double error_epoch = 0.0;
    int epoch = 0;

    initw(w);
    n_of_set = getdata(x, desired);
    printf("데이터셋 개수:%d\n", n_of_set);

    neural_forward(x, w, n_of_set, output);
    for (i = 0; i < n_of_set; ++i)
    {
        printf("%d", i);
        for (j = 1; j <= NUM_INPUT_NODE; ++j)
            printf("%lf", x[i][j]);
        O = forward(x[i], w);
        printf("=%lf\n", output[i]);
    }

    printf("\n**********학습시작***********\n");
    for (epoch = 0; epoch < MAX_EPOCH; epoch++)
    {
        err = neural_learning(x, w, n_of_set, desired);
        printf("epoch=%d avg err=%lf\n", epoch, err);
        if (err == 0.0)
            ;
        break;
    }
    printf("**********학습종료***********\n");

    neural_forward(x, w, n_of_set, output);
    for (i = 0; i < n_of_set; ++i)
    {
        printf("%d", i);
        for (j = 1; j <= NUM_INPUT_NODE; ++j)
            printf("%lf", x[i][j]);
        printf("=%lf\n", output[i]);
    }

    printf("bias_weight=%lf\n", w[0]);
    for (j = 1; j <= NUM_INPUT_NODE; ++j)
        printf("weight=%d = %lf", j, w[j]);
    printf("\n");
    return 0;
}

void initw(double weight[NUM_INPUT_NODE + 1])
{
    srand(time(NULL));
    for (int i = 0; i < MAXNUM_NUM_DATA_SET + 1; i++)
        weight[i] = (double)rand() / RAND_MAX - 0.5;
    // weight[0] = -0.5;
    // weight[1] = 1;
    // weight[2] = 1;
}

int getdata(double in[][NUM_INPUT_NODE + 1], double desired[])
{
    int n_of_set = 0;
    int j = 0;

    while (scanf("%lf", &in[n_of_set][j]) != EOF)
    {
        ++j;
        if (j > NUM_INPUT_NODE)
        {
            j = 1;
            n_of_set++;
        }
        if (n_of_set == MAXNUM_NUM_DATA_SET)
            break;
    }
    // 바이어스
    in[0][0] = 1;
    in[0][1] = 0;
    in[0][2] = 0;
    desired[0] = 1;
    in[1][0] = 1;
    in[1][1] = 0;
    in[1][2] = 1;
    desired[1] = 1;
    in[2][0] = 1;
    in[2][1] = 1;
    in[2][2] = 0;
    desired[2] = 0;
    in[3][0] = 1;
    in[3][1] = 1;
    in[3][2] = 1;
    desired[3] = 0;
    n_of_set = 4;
    return n_of_set;
}

double forward(double x[NUM_INPUT_NODE + 1],
               double w[NUM_INPUT_NODE + 1])
{
    int i;
    double weightsum;
    double result;

    weightsum = 0.0;
    weightsum += (1 * w[0]);
    for (i = 1; i <= NUM_INPUT_NODE; i++)
        weightsum += x[i] * w[i];

    result = activationfunc(weightsum);
    return result;
}

double activationfunc(double sum)
{
    if (sum >= 0)
        return 1.0;
    else
        return 0.0;
    // return(s>+0)?1.0 : 0.0;
    //  return 1.0/(1.0+exp(-u));
}

void neural_forward(double in[][NUM_INPUT_NODE + 1],
                    double w[NUM_INPUT_NODE + 1], int n_of_set,
                    double out[MAXNUM_NUM_DATA_SET])
{
    int i;
    for (i = 0; i < n_of_set; i++)
        out[i] = forward(in[i], w);
}

double neural_learning(double in[][NUM_INPUT_NODE + 1],
                       double w[NUM_INPUT_NODE + 1], int n_of_set,
                       double dout[MAXNUM_NUM_DATA_SET])
{
    int i, j;
    double out;
    double err;
    double error_sum;

    for (j = 0; j < n_of_set; j++)
    {
        out = forward(in[j], w);
        err = dout[j] - out;
        w[0] = w[0] + LEARNING_RATE * in[j][i] * err;
        for (i = 1; i <= NUM_INPUT_NODE; i++)
            w[i] = w[i] + LEARNING_RATE * in[j][i] * err;
    }

    error_sum = 0.0;
    for (j = 0; j < n_of_set; j++)
    {
        out = forward(in[j], w);
        err = (dout[j] - out) * (dout[j] - out);
        error_sum += fabs(err);
    }
    return error_sum / n_of_set;
}
