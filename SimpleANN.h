// SimpleANN (Simple Artificial Neural Network)
// Copyright © 2021, 2022 Ilkka Pokkinen
// Released under MIT license


#pragma once
#include <algorithm>


namespace sann
{
    struct NeuralNetwork
    {
        unsigned nLayers;
        float momentumMultiplier;
        float learningRate;
        float maxMomentum;
        unsigned* layerSizes;
        float** values;
        float** errors;
        float** weights;
        float** dWeights;
        float** weightMomentums;
        float** biases;
        float** dBiases;
        float** biasMomentums;
    };

    inline void allocNN(NeuralNetwork& neuralNetwork, bool allocMomentum)
    {
        const unsigned nLayers = neuralNetwork.nLayers;

        neuralNetwork.values = new float* [nLayers];
        neuralNetwork.errors = new float* [nLayers];
        neuralNetwork.weights = new float* [nLayers];
        neuralNetwork.dWeights = new float* [nLayers];
        neuralNetwork.biases = new float* [nLayers];
        neuralNetwork.dBiases = new float* [nLayers];
        if (allocMomentum)
        {
            neuralNetwork.weightMomentums = new float* [nLayers];
            neuralNetwork.biasMomentums = new float* [nLayers];
        }
        else
        {
            neuralNetwork.weightMomentums = nullptr;
            neuralNetwork.biasMomentums = nullptr;
        }

        for (unsigned i = 0; i < nLayers; ++i)
            neuralNetwork.values[i] = new float[neuralNetwork.layerSizes[i]];
        for (unsigned i = 1; i < nLayers; ++i)
        {
            neuralNetwork.errors[i] = new float[neuralNetwork.layerSizes[i]];
            neuralNetwork.weights[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            neuralNetwork.dWeights[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            neuralNetwork.biases[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            neuralNetwork.dBiases[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            if (allocMomentum)
            {
                neuralNetwork.weightMomentums[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
                neuralNetwork.biasMomentums[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            }
        }
    }

    inline void deallocNN(NeuralNetwork& neuralNetwork, unsigned* layerSizes)
    {
        for (unsigned i = 0; i < neuralNetwork.nLayers; ++i)
            delete[](neuralNetwork.values[i]);
        for (unsigned i = 1; i < neuralNetwork.nLayers; ++i)
        {
            delete[](neuralNetwork.errors[i]);
            delete[](neuralNetwork.weights[i]);
            delete[](neuralNetwork.dWeights[i]);
            delete[](neuralNetwork.weightMomentums[i]);
            delete[](neuralNetwork.biases[i]);
            delete[](neuralNetwork.dBiases[i]);
            delete[](neuralNetwork.biasMomentums[i]);
        }
    }

    inline void propagateForward(NeuralNetwork& nn, float(*actFuncs[])(float))
    {
        for (unsigned layerIndex = 1; layerIndex < nn.nLayers; ++layerIndex)
        {
            for (unsigned i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                float temp = 0.0f;
                unsigned prevLayerSize = nn.layerSizes[layerIndex - 1];
                for (unsigned j = 0; j < prevLayerSize; ++j)
                    temp += nn.weights[layerIndex][i * prevLayerSize + j];
                nn.values[layerIndex][i] = actFuncs[layerIndex](temp + nn.biases[layerIndex][i]);
            }
        }
    }

    inline void propagateBackwards(NeuralNetwork& nn, float(*derFuncs[])(float), const float* label)
    {
        // Calculate errors
        // output layer
        for (unsigned i = 0; i < nn.layerSizes[nn.nLayers - 1]; ++i)
            nn.errors[nn.nLayers - 1][i] = label[i] - nn.values[nn.nLayers - 1][i];

        // hidden layers
        for (unsigned layerIndex = nn.nLayers - 2; layerIndex > 0; --layerIndex)
        {
            for (unsigned i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                nn.errors[layerIndex][i] = 0.0f;
                for (unsigned j = 0; j < nn.layerSizes[layerIndex + 1]; ++j)
                    nn.errors[layerIndex][i] += nn.errors[layerIndex + 1][j] * nn.weights[layerIndex + 1][j * nn.layerSizes[layerIndex + 1] + i];
            }
        }

        // Calculate error derivatives
        for (unsigned layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (unsigned i = 0; i < nn.layerSizes[layerIndex]; ++i)
                nn.errors[layerIndex][i] *= derFuncs[layerIndex](nn.values[layerIndex][i]);
        }

        // Cumulate per weight and bias changes
        for (unsigned layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (unsigned i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                nn.dBiases[layerIndex][i] += nn.errors[layerIndex][i];
                for (int j = 0; j < nn.layerSizes[layerIndex - 1]; ++j)
                    nn.dWeights[layerIndex][i * nn.layerSizes[layerIndex - 1] + j] += nn.errors[layerIndex][i] * nn.values[layerIndex - 1][j];
            }
        }
    }

    inline void update(NeuralNetwork& nn, unsigned epochs)
    {
        unsigned weightIndex;
        for (int layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (int i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                if (nn.weightMomentums != nullptr)
                {
                    nn.biases[layerIndex][i] += nn.learningRate * nn.dBiases[layerIndex][i] / epochs + nn.biasMomentums[layerIndex][i];
                    nn.biasMomentums[layerIndex][i] = nn.momentumMultiplier * nn.biasMomentums[layerIndex][i] + nn.momentumMultiplier * nn.dBiases[layerIndex][i];
                    nn.biasMomentums[layerIndex][i] = std::clamp(nn.biasMomentums[layerIndex][i], -nn.maxMomentum, nn.maxMomentum);
                }
                else
                {
                    nn.biases[layerIndex][i] += nn.learningRate * nn.dBiases[layerIndex][i] / epochs;
                }
                nn.dBiases[layerIndex][i] = 0.0f;
                for (unsigned j = 0; nn.layerSizes[layerIndex - 1]; ++j)
                {
                    weightIndex = i * nn.layerSizes[layerIndex - 1] + j;
                    if (nn.weightMomentums != nullptr)
                    {
                        nn.weights[layerIndex][weightIndex] += nn.learningRate * nn.dWeights[layerIndex][weightIndex] / epochs + nn.weightMomentums[layerIndex][weightIndex];
                        nn.weightMomentums[layerIndex][weightIndex] = nn.momentumMultiplier * nn.weightMomentums[layerIndex][weightIndex] + nn.momentumMultiplier * nn.dWeights[layerIndex][weightIndex];
                        nn.weightMomentums[layerIndex][weightIndex] = std::clamp(nn.weightMomentums[layerIndex][weightIndex], -nn.maxMomentum, nn.maxMomentum);
                    }
                    else
                    {
                        nn.weights[layerIndex][weightIndex] += nn.learningRate * nn.dWeights[layerIndex][weightIndex] / epochs;
                    }
                    nn.dWeights[layerIndex][weightIndex] = 0.0f;
                }
            }
        }
    }

    inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

    inline float dSigmoid(float x)
    {
        return x * (1.0f - x);
    }

    inline float relu(float x)
    {
        if (x)
            return x;
        return 0.0f;
    }

    inline float dRelu(float x)
    {
        if (x)
            return 1.0f;
        return 0.0f;
    }

    inline float leakyRelu(float x)
    {
        if (x)
            return x;
        return 0.01f * x;
    }

    inline float dLeakyRelu(float x)
    {
        if (x)
            return 1.0f;
        return 0.01f;
    }

    inline float hyperbolicTanh(float x)
    {
        return tanhf(x);
    }

    inline float dHyperbolicTanh(float x)
    {
        return 1.0f - x * x;
    }
}