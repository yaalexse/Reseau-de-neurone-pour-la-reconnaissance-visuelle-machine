/*
Yadrin Alexei, 2023
https://github.com/yaalexse/NN-for-Mnist-in-C
Require the "mnist.h" header made by Takafumi Hoiruchi
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "mnist.h"

// no neuron/layer struct
// no matrix multiplication function
// only 8 function


float Sig(float x) { return 1.0 / (1.0 + exp(-x)); }
float dSig(float x) { return Sig(x) * (1.0 - Sig(x)); }

void softmax(float *in, float *out, int n)
{
    float max = -INFINITY;
    for (int i = 0; i < n; i++)
    {
        if (in[i] > max)
            max = in[i];
    }

    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += exp(in[i] - max);

    for (int i = 0; i < n; i++)
        out[i] = exp(in[i] - max) / sum;
}

int argmax(float *x, int n)
{
    float max = -INFINITY;
    int max_i = -1;
    for (int i = 0; i < n; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
            max_i = i;
        }
    }
    return max_i;
}

void forward_prop(bool test, int *lay, float **act, float **actp, float **wei, int l, int len_lay)
{
    int i, j, k;
    float som;
    if (test)
        for (i = 0; i < *(lay); i++)
        {
            *(*(act) + i) = (float)test_image[l][i];
            *(*(actp) + i) = (float)test_image[l][i];
        }
    else
        for (i = 0; i < *(lay); i++)
        {
            *(*(act) + i) = (float)train_image[l][i];
            *(*(actp) + i) = (float)train_image[l][i];
        }
    for (k = 1; k < len_lay; k++)
    {
        for (j = 0; j < *(lay + k); j++)
        {
            som = 0.;
            for (i = 0; i < *(lay + k - 1); i++)
            {
                som = som + *(*(act + k - 1) + i) * *(*(wei + k - 1) + i * *(lay + k) + j);
            }
            *(*(act + k) + j) = Sig(som);
            *(*(actp + k) + j) = som;
        }
    }
    softmax(actp[len_lay - 1], act[len_lay - 1],
            lay[len_lay - 1]);
}

void back_prop(int *lay, float **act, float **actp, float **wei, float **delta, int train_label, int len_lay)
{
    int i, j, k;
    float som;
    for (j = 0; j < *(lay + len_lay - 1); j++)
    {
        *(*(delta + len_lay - 1) + j) = *(*(act + len_lay - 1) + j) - (j == train_label);
    }
    for (k = len_lay - 2; k > 0; k--)
    {
        for (j = 0; j < *(lay + k); j++)
        {
            som = 0.0;
            for (i = 0; i < *(lay + k + 1); i++)
            {
                som += *(*(delta + k + 1) + i) * *(*(wei + k) + j * *(lay + k + 1) + i);
            }
            *(*(delta + k) + j) = som * dSig(*(*(actp + k) + j));
        }
    }
}

float rn()
{
    float a;
    a = 2 * (float)rand() / (float)RAND_MAX - 1.0;
    return a;
}

float unif(float high)
{
    return high * rn();
}

int main()
{
    //---net struct input---
    int info[4] = {784, 200, 50, 10}, len_info = 4;

    //---memory allocation---
    int i, j, k, l, m, p;
    int len_lay = len_info;
    int *lay = (int *)malloc(sizeof(int) * len_lay);
    float **delta = (float **)malloc(sizeof(float *) * len_lay);
    float **act = (float **)malloc(sizeof(float *) * len_lay);
    float **actp = (float **)malloc(sizeof(float *) * len_lay);
    for (i = 0; i < len_lay; i++)
    {
        *(lay + i) = info[i];
        *(act + i) = (float *)malloc(sizeof(float) * *(lay + i));
        *(delta + i) = (float *)malloc(sizeof(float) * *(lay + i));
        *(actp + i) = (float *)malloc(sizeof(float) * *(lay + i));
    }
    float **wei = (float **)malloc(sizeof(float *) * len_lay);
    for (i = 0; i < len_lay - 1; i++)
    {
        *(wei + i) = (float *)malloc(sizeof(float) * (*(lay + i)) * (*(lay + i + 1)));
    }
    srand(time(NULL));
    printf("\tmemory allocation: ok\t");

    //---initialization---
    for (i = 0; i < len_lay - 1; i++)
    {
        for (j = 0; j < *(lay + i); j++)
        {
            for (k = 0; k < *(lay + i + 1); k++)
            {
                *(*(wei + i) + *(lay + i + 1) * j + k) = unif(1.0 / (sqrt(*(lay + i))));
            }
        }
    }
    printf("\tinitialization: ok");

    //---trainnig the net---
    load_mnist();
    int nb_train = 60000, res;
    float lr = 0.01;
    for (k = 0; k < nb_train; k++)
    {
        forward_prop(false, lay, act, actp, wei, k, len_lay);
        back_prop(lay, act, actp, wei, delta, train_label[k], len_lay);
        printf("label[%d]:%d\n", k, train_label[k]);
        for (i = 0; i < *(lay + len_lay - 1); i++)
        {
            printf(" ,%f   ", *(*(act + len_lay - 1) + i));
        }

        res = argmax(act[len_lay - 1], lay[len_lay - 1]);
        printf("\nresult:%d\n", res);

        for (i = 0; i < len_lay - 1; i++)
        {
            for (m = 0; m < *(lay + i + 1); m++)
            {
                for (j = 0; j < *(lay + i); j++)
                {
                    *(*(wei + i) + j * *(lay + i + 1) + m) -= lr * *(*(delta + i + 1) + m) * *(*(actp + i) + j);
                }
            }
        }
    }

    //---testing the net---
    printf("test phase");
    int success = 0, nb_test = 10000;
    for (k = 0; k < nb_test; k++)
    {
        forward_prop(true, lay, act, actp, wei, k, len_lay);
        res = argmax(*(act + len_lay - 1), *(lay + len_lay - 1));
        success += (res == test_label[k]);
        printf("label[%d]:%d\n", k, test_label[k]);
        for (i = 0; i < *(lay + len_lay - 1); i++)
        {
            printf(" ,%f   ", *(*(act + len_lay - 1) + i));
        }
        printf("\nresult:%d\n", res);
        printf("success:%d", success);
    }
    printf("success rate: %f\n", 100. * success / nb_test);

    return 0;
}
