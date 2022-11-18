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

	struct NeuralNetworkAllocInfo
	{
		unsigned nLayers;
		unsigned* layerSizes;
		float momentumMultiplier;
		float maxMomentum;
		float leaningRate;
	};

	struct NeuralNetwork
	{
		unsigned nLayers;
		float learningRate;
		float momentumMultiplier;
		float maxMomentum;
		float epochError;
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

	inline void allocNeuralNetwork(NeuralNetwork& nn, const NeuralNetworkAllocInfo& createInfo)
	{
		const unsigned nLayers = createInfo.nLayers;

		nn.nLayers = nLayers;
		nn.learningRate = createInfo.leaningRate;
		nn.momentumMultiplier = createInfo.momentumMultiplier;
		nn.maxMomentum = createInfo.maxMomentum;
		nn.layerSizes = new unsigned[nLayers];
		nn.zValues = new float* [nLayers];
		nn.aValues = new float* [nLayers];
		nn.errors = new float* [nLayers];
		nn.weights = new float* [nLayers];
		nn.dWeights = new float* [nLayers];
		nn.weightMomentums = new float* [nLayers];
		nn.biases = new float* [nLayers];
		nn.dBiases = new float* [nLayers];
		nn.biasMomentums = new float* [nLayers];

		for (unsigned i = 0; i < nLayers; ++i)
		{
			nn.layerSizes[i] = createInfo.layerSizes[i];
			nn.zValues[i] = new float[createInfo.layerSizes[i]] { 0.0f };
			nn.aValues[i] = new float[createInfo.layerSizes[i]] { 0.0f };
		}

		for (unsigned i = 1; i < nLayers; ++i)
		{
			nn.errors[i] = new float[createInfo.layerSizes[i]];
			nn.weights[i] = new float[createInfo.layerSizes[i] * createInfo.layerSizes[i - 1]];
			nn.dWeights[i] = new float[createInfo.layerSizes[i] * createInfo.layerSizes[i - 1]] { 0.0f };
			nn.biases[i] = new float[createInfo.layerSizes[i]];
			nn.dBiases[i] = new float[createInfo.layerSizes[i]] { 0.0f };
			nn.weightMomentums[i] = new float[createInfo.layerSizes[i] * createInfo.layerSizes[i - 1]] { 0.0f };
			nn.biasMomentums[i] = new float[createInfo.layerSizes[i]] { 0.0f };
		}
	}

	inline void deallocNeuralNetwork(NeuralNetwork& nn)
	{
		for (unsigned i = 0; i < nn.nLayers; ++i)
		{
			delete[](nn.zValues[i]);
			delete[](nn.aValues[i]);
		}
		for (unsigned i = 1; i < nn.nLayers; ++i)
		{
			delete[](nn.errors[i]);
			delete[](nn.weights[i]);
			delete[](nn.dWeights[i]);
			delete[](nn.biases[i]);
			delete[](nn.dBiases[i]);
			delete[](nn.weightMomentums[i]);
			delete[](nn.biasMomentums[i]);
		}
		delete[](nn.layerSizes);
		delete[](nn.zValues);
		delete[](nn.aValues);
		delete[](nn.errors);
		delete[](nn.weights);
		delete[](nn.dWeights);
		delete[](nn.weightMomentums);
		delete[](nn.biases);
		delete[](nn.dBiases);
		delete[](nn.biasMomentums);
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
				sum += nn.biases[layer][node];
				nn.zValues[layer][node] = sum;
				nn.aValues[layer][node] = stepFuncs[layer].activation(sum);
			}
		}
	}

	inline void propagateBackwards(NeuralNetwork& nn, const StepFunction stepFuncs[], const float* label)
	{
		// network error & error at output layer
		nn.epochError = 0.0f;
		for (unsigned i = 0; i < nn.layerSizes[nn.nLayers - 1]; ++i)
		{
			float distance = label[i] - nn.aValues[nn.nLayers - 1][i];
			float stepFuncDerivative = stepFuncs[nn.nLayers - 1].derivative(nn.aValues[nn.nLayers - 1][i]);
			nn.epochError += 0.5f * distance * distance;
			nn.errors[nn.nLayers - 1][i] = stepFuncDerivative * -distance;

			for (unsigned prevNode = 0; prevNode < nn.layerSizes[nn.nLayers - 2]; ++prevNode)
				nn.dWeights[nn.nLayers - 1][i * nn.layerSizes[nn.nLayers - 2] + prevNode]
				+= nn.aValues[nn.nLayers - 2][prevNode] * nn.errors[nn.nLayers - 1][i];
			nn.dBiases[nn.nLayers - 1][i] += nn.errors[nn.nLayers - 1][i];
		}

		// hidden layers
		for (unsigned layerIndex = nn.nLayers - 2; layerIndex > 0; --layerIndex)
		{
			for (unsigned node = 0; node < nn.layerSizes[layerIndex]; ++node)
			{
				float stepFuncDerivative = stepFuncs[layerIndex].derivative(nn.aValues[layerIndex][node]);
				float totalErrorIn = 0.0f;
				for (unsigned nodeNext = 0; nodeNext < nn.layerSizes[layerIndex + 1]; ++nodeNext)
					totalErrorIn += nn.weights[layerIndex + 1][nodeNext * nn.layerSizes[layerIndex] + node] * nn.errors[layerIndex + 1][nodeNext];
				nn.errors[layerIndex][node] = stepFuncDerivative * totalErrorIn;

				for (unsigned prevNode = 0; prevNode < nn.layerSizes[layerIndex - 1]; ++prevNode)
					nn.dWeights[layerIndex][node * nn.layerSizes[layerIndex - 1] + prevNode]
					+= nn.aValues[layerIndex - 1][prevNode] * nn.errors[layerIndex][node];
				nn.dBiases[layerIndex][node] += nn.errors[layerIndex][node];
			}
		}
	}

	inline float calculateLoss(const NeuralNetwork& nn, const float* label)
	{
		float loss = 0.0f;
		for (unsigned i = 0; i < nn.layerSizes[nn.nLayers - 1]; ++i)
		{
			float distance = label[i] - nn.aValues[nn.nLayers - 1][i];
			loss += 0.5f * distance * distance;
		}
		return loss;
	}

	inline void update(NeuralNetwork& nn, unsigned epochs)
	{
		for (unsigned layerIndex = nn.nLayers - 1; layerIndex > 0; --layerIndex)
		{
			for (unsigned i = 0; i < nn.layerSizes[layerIndex]; ++i)
			{
				nn.biases[layerIndex][i] -= nn.learningRate * nn.dBiases[layerIndex][i] / epochs + nn.biasMomentums[layerIndex][i];
				nn.biasMomentums[layerIndex][i] = nn.momentumMultiplier * nn.biasMomentums[layerIndex][i] + nn.momentumMultiplier * nn.dBiases[layerIndex][i];
				nn.biasMomentums[layerIndex][i] = std::clamp(nn.biasMomentums[layerIndex][i], -nn.maxMomentum, nn.maxMomentum);
				nn.dBiases[layerIndex][i] = 0.0f;
				for (unsigned j = 0; j < nn.layerSizes[layerIndex - 1]; ++j)
				{
					unsigned weightIndex = i * nn.layerSizes[layerIndex - 1] + j;
					nn.weights[layerIndex][weightIndex] -= nn.learningRate * nn.dWeights[layerIndex][weightIndex] / epochs + nn.weightMomentums[layerIndex][weightIndex];
					nn.weightMomentums[layerIndex][weightIndex] = nn.momentumMultiplier * nn.weightMomentums[layerIndex][weightIndex] + nn.momentumMultiplier * nn.dWeights[layerIndex][weightIndex];
					nn.weightMomentums[layerIndex][weightIndex] = std::clamp(nn.weightMomentums[layerIndex][weightIndex], -nn.maxMomentum, nn.maxMomentum);
					nn.dWeights[layerIndex][weightIndex] = 0.0f;
				}
			}
		}
	}

	// Step functions
	inline float linear(float x) { return x; }

	inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

	inline float dSigmoid(float x) { return x * (1.0f - x); }

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

	inline float hyperbolicTanh(float x) { return tanhf(x); }

	inline float dHyperbolicTanh(float x) { return 1.0f - x * x; }
}