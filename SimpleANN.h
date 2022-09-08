// SimpleANN (Simple Artificial Neural Network)
// Copyright © 2021, 2022 Ilkka Pokkinen
// Released under MIT license


#pragma once
#include <algorithm>


namespace sann
{
    struct StepFunction
    {
        float (*activation)(float);
        float (*derivative)(float);
    };

    struct NeuralNetwork
    {
        unsigned nLayers;
        float momentumMultiplier;
        float learningRate;
        float maxMomentum;
        float mse;
        unsigned* layerSizes;
        float** zValues;
        float** aValues;
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

        neuralNetwork.zValues = new float* [nLayers];
        neuralNetwork.aValues = new float* [nLayers];
        neuralNetwork.errors = new float* [nLayers];
        neuralNetwork.weights = new float* [nLayers];
        neuralNetwork.dWeights = new float* [nLayers];
        neuralNetwork.biases = new float* [nLayers];
        neuralNetwork.dBiases = new float* [nLayers];
        if (nLayers > 0)
        {
            neuralNetwork.errors[0] = nullptr;
            neuralNetwork.biases[0] = nullptr;
            neuralNetwork.dBiases[0] = nullptr;
            neuralNetwork.weights[0] = nullptr;
            neuralNetwork.dWeights[0] = nullptr;
        }
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
        {
            neuralNetwork.zValues[i] = new float[neuralNetwork.layerSizes[i]] { 0.0f };
            neuralNetwork.aValues[i] = new float[neuralNetwork.layerSizes[i]] { 0.0f };
        }
        for (unsigned i = 1; i < nLayers; ++i)
        {
            neuralNetwork.errors[i] = new float[neuralNetwork.layerSizes[i]];
            neuralNetwork.weights[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]];
            neuralNetwork.dWeights[i] = new float[neuralNetwork.layerSizes[i] * neuralNetwork.layerSizes[i - 1]] { 0.0f };
            neuralNetwork.biases[i] = new float[neuralNetwork.layerSizes[i]];
            neuralNetwork.dBiases[i] = new float[neuralNetwork.layerSizes[i]] { 0.0f };
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
        {
            delete[](neuralNetwork.zValues[i]);
            delete[](neuralNetwork.aValues[i]);
        }
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

    inline void propagateForward(NeuralNetwork& nn, const StepFunction stepFuncs[])
    {
        for (unsigned layer = 1; layer < nn.nLayers; ++layer)
        {
            for (unsigned node = 0; node < nn.layerSizes[layer]; ++node)
            {
                float sum = 0.0f;
                for (unsigned prevNode = 0; prevNode < nn.layerSizes[layer - 1]; ++prevNode)
                    sum += nn.aValues[layer - 1][prevNode] * nn.weights[layer][node * nn.layerSizes[layer - 1] + prevNode];
                nn.zValues[layer][node] = sum;
                nn.aValues[layer][node] = stepFuncs[layer].activation(sum);
            }
        }
    }

    inline void propagateBackwards(NeuralNetwork& nn, const StepFunction stepFuncs[], const float* label)
    {
        // mean square error & error at output layer
        nn.mse = 0.0f;
        float dmse = 0.0f;
        for (unsigned i = 0; i < nn.layerSizes[nn.nLayers - 1]; ++i)
        {
            float dist = label[i] - nn.aValues[nn.nLayers - 1][i];
            nn.mse += dist * dist;
            dmse += 2 * dist;
        }

        // output layer
        for (unsigned node = 0; node < nn.layerSizes[nn.nLayers - 1]; ++node)
            nn.errors[nn.nLayers - 1][node] = stepFuncs[nn.nLayers - 1].derivative(nn.zValues[nn.nLayers - 1][node]) * dmse;

        // hidden layers
        for (unsigned layerIndex = nn.nLayers - 2; layerIndex > 0; --layerIndex)
        {
            for (unsigned node = 0; node < nn.layerSizes[layerIndex]; ++node)
            {
                float errorSum = 0.0f;
                for (unsigned nodeNext = 0; nodeNext < nn.layerSizes[layerIndex + 1]; ++nodeNext)
                    errorSum += nn.weights[layerIndex + 1][nodeNext * nn.layerSizes[layerIndex] + node] * nn.errors[layerIndex + 1][nodeNext];
                nn.errors[layerIndex][node] = stepFuncs[layerIndex].derivative(nn.zValues[layerIndex][node]) * errorSum;
            }
        }

        // cumulate weight and bias changes
        for (unsigned layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (unsigned node = 0; node < nn.layerSizes[layerIndex]; ++node)
            {
                nn.dBiases[layerIndex][node] += nn.errors[layerIndex][node];
                for (int nodePrev = 0; nodePrev < nn.layerSizes[layerIndex - 1]; ++nodePrev)
                    nn.dWeights[layerIndex][node * nn.layerSizes[layerIndex - 1] + nodePrev] += nn.aValues[layerIndex - 1][nodePrev] * nn.errors[layerIndex][node];
            }
        }
    }

    inline void update(NeuralNetwork& nn, unsigned epochs)
    {
        if (nn.weightMomentums == nullptr)
            updateNoMomentum(nn, epochs);
        else
            updateWithMomentum(nn, epochs);
    }

    inline void updateNoMomentum(NeuralNetwork& nn, unsigned epochs)
    {
        unsigned weightIndex;
        for (int layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (int i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                nn.biases[layerIndex][i] -= nn.learningRate * nn.dBiases[layerIndex][i] / epochs;
                nn.dBiases[layerIndex][i] = 0.0f;
                for (unsigned j = 0; j < nn.layerSizes[layerIndex - 1]; ++j)
                {
                    weightIndex = i * nn.layerSizes[layerIndex - 1] + j;
                    nn.weights[layerIndex][weightIndex] -= nn.learningRate * nn.dWeights[layerIndex][weightIndex] / epochs;
                    nn.dWeights[layerIndex][weightIndex] = 0.0f;
                }
            }
        }
    }

    inline void updateWithMomentum(NeuralNetwork& nn, unsigned epochs)
    {
        unsigned weightIndex;
        for (int layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
        {
            for (int i = 0; i < nn.layerSizes[layerIndex]; ++i)
            {
                nn.biases[layerIndex][i] -= nn.learningRate * nn.dBiases[layerIndex][i] / epochs + nn.biasMomentums[layerIndex][i];
                nn.biasMomentums[layerIndex][i] = nn.momentumMultiplier * nn.biasMomentums[layerIndex][i] + nn.momentumMultiplier * nn.dBiases[layerIndex][i];
                nn.biasMomentums[layerIndex][i] = std::clamp(nn.biasMomentums[layerIndex][i], -nn.maxMomentum, nn.maxMomentum);
                nn.dBiases[layerIndex][i] = 0.0f;
                for (unsigned j = 0; j < nn.layerSizes[layerIndex - 1]; ++j)
                {
                    weightIndex = i * nn.layerSizes[layerIndex - 1] + j;
                    nn.weights[layerIndex][weightIndex] -= nn.learningRate * nn.dWeights[layerIndex][weightIndex] / epochs + nn.weightMomentums[layerIndex][weightIndex];
                    nn.weightMomentums[layerIndex][weightIndex] = nn.momentumMultiplier * nn.weightMomentums[layerIndex][weightIndex] + nn.momentumMultiplier * nn.dWeights[layerIndex][weightIndex];
                    nn.weightMomentums[layerIndex][weightIndex] = std::clamp(nn.weightMomentums[layerIndex][weightIndex], -nn.maxMomentum, nn.maxMomentum);
                    nn.dWeights[layerIndex][weightIndex] = 0.0f;
                }
            }
        }
    }

    inline float linear(float x)
    {
        return x;
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