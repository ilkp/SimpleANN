// SimpleANN (Simple Artificial Neural Network)
// Copyright © 2021, 2022 Ilkka Pokkinen
// Released under MIT license

#pragma once

#include <math.h>
#include <algorithm>

namespace sann
{
	// Sigmoid function. Outputs value between 0.0f and 1.0f. Slow to calculate so recommended to use for only the output layer.
	inline float sigmoidAct(float x) { return 1.0f / (1.0f + expf(-x)); }
	inline float sigmoidDer(float x) { return x * (1.0f - x); }
	// Rectified linear unit function. Outputs 0 when x < 0, and outputs x when x >= 0.
	inline float reluAct(float x) { return x < 0.0f ? 0.0f : x; }
	inline float reluDer(float x) { return x < 0.0f ? 0.0f : 1.0f; }
	// Leaky rectified linear unit. Outputs 0.01 * x when x < 0, and outputs x when x >= 0.
	inline float leakyReluAct(float x) { return x < 0.0f ? 0.01f * x : x; }
	inline float leakyReluDer(float x) { return x < 0.0f ? 0.01f : 1.0f; }
	// Hyperbolic tangent function. Outputs tanhf(x) of math.h
	inline float hyperbolicTanhAct(float x) { return tanhf(x); }
	inline float hyperbolicTanhDer(float x) { return 1.0f - x * x; }
	
	struct SANN
	{
		float (**activationFuncs)(float);
		float (**derivationFuncs)(float);
		float leaningRate;
		float momentumMultiplier;
		float maxMomentum;
		unsigned nLayers;
		unsigned* layerSizes;
		float** nodeValues;
		float** weights;
		float** deltaWeights;
		float** weightsMomentums;
		float** biases;
		float** deltaBiases;
		float** biasMomentums;
		float** errors;
	};
	
	inline void propagateForward(SANN& sann, const float* input)
	{
		std::copy(input, input + sann.layerSizes[0], sann.nodeValues);
		for (unsigned layerIndex = 1; layerIndex < sann.nLayers; ++layerIndex)
		{
			for (unsigned i = 0; i < sann.layerSizes[layerIndex]; ++i)
			{
				float temp = 0.0f;
				unsigned prevLayerSize = sann.layerSizes[layerIndex - 1];
				for (unsigned j = 0; j < prevLayerSize; ++j)
					temp += sann.weights[layerIndex][i * prevLayerSize + j];
				sann.nodeValues[layerIndex][i] = sann.activationFuncs[layerIndex](temp + sann.biases[layerIndex][i]);
			}
		}
	}
	
	inline void propagateBackward(SANN& sann, const float* label)
	{
		// Calculate errors
		// output layer
		for (unsigned i = 0; i < sann.layerSizes[sann.nLayers - 1]; ++i)
			sann.errors[sann.nLayers - 1][i] = label[i] - sann.nodeValues[sann.nLayers - 1][i];
		
		// hidden layers
		for (unsigned layerIndex = sann.nLayers - 2; layerIndex > 0; --layerIndex)
		{
			for (unsigned i = 0; i < sann.layerSizes[layerIndex]; ++i)
			{
				sann.errors[layerIndex][i] = 0.0f;
				for (unsigned j = 0; j < sann.layerSizes[layerIndex + 1]; ++j)
					sann.errors[layerIndex][i] += sann.errors[layerIndex + 1][j] * sann.weights[layerIndex + 1][j * sann.layerSizes[layerIndex + 1] + i];
			}
		}
		
		// Calculate error derivatives
		for (unsigned layerIndex = sann.nLayers - 1; layerIndex > 0; --layerIndex)
		{
			for (unsigned i = 0; i < sann.layerSizes[layerIndex]; ++i)
				sann.errors[layerIndex][i] *= sann.derivationFuncs[layerIndex](sann.nodeValues[layerIndex][i]);
		}
		
		// Cumulate per weight and bias changes
		for (unsigned layerIndex = sann.nLayers - 1; layerIndex > 0; --layerIndex)
		{
			for (unsigned i = 0; i < sann.layerSizes[layerIndex]; ++i)
			{
				sann.deltaBiases[layerIndex][i] += sann.errors[layerIndex];
				for (int j = 0; j < sann.layerSizes[layerIndex - 1]; ++j)
					sann.deltaWeights[i * sann.layerSizes[layerIndex - 1] + j] += sann.errors[layerIndex][i] * sann.nodeValues[layerIndex - 1][j];
			}
		}
	}
	
	inline void update(SANN& sann, unsigned epochs)
	{
		unsigned weightIndex;
		for (int layerIndex = sann.nLayers - 1; layerIndex > 0; --layerIndex)
		{
			for (int i = 0; i < sann.layerSizes[layerIndex]; ++i)
			{
				if (sann.momentumMultiplier != 0.0f)
				{
					sann.biases[layerIndex][i] += sann.leaningRate * sann.deltaBiases[layerIndex][i] / epochs + sann.biasMomentums[layerIndex][i];
					sann.biasMomentums[layerIndex][i] = sann.momentumMultiplier * sann.biasMomentums[layerIndex][i] + sann.momentumMultiplier * sann.deltaBiases[layerIndex][i];
					sann.biasMomentums[layerIndex][i] = std::clamp(sann.biasMomentums[layerIndex][i], -sann.maxMomentum, sann.maxMomentum);
				}
				else
				{
					sann.biases[layerIndex][i] += sann.leaningRate * sann.deltaBiases[layerIndex][i] / epochs;
				}
				sann.deltaBiases[layerIndex][i] = 0.0f;
				for (unsigned j = 0; sann.layerSizes[layerIndex - 1]; ++j)
				{
					weightIndex = i * sann.layerSizes[layerIndex - 1] + j;
					if (sann.momentumMultiplier != 0.0f)
					{
						sann.weights[layerIndex][weightIndex] += sann.leaningRate * sann.deltaWeights[layerIndex][weightIndex] / epochs + sann.weightsMomentums[layerIndex][weightIndex];
						sann.weightsMomentums[layerIndex][weightIndex] = sann.momentumMultiplier * sann.weightsMomentums[layerIndex][weightIndex] + sann.momentumMultiplier * sann.deltaWeights[layerIndex][weightIndex];
						sann.weightsMomentums[layerIndex][weightIndex] = std::clamp(sann.weightsMomentums[layerIndex][weightIndex], -sann.maxMomentum, sann.maxMomentum);
					}
					else
					{
						sann.weights[layerIndex][weightIndex] += sann.leaningRate * sann.deltaWeights[layerIndex][weightIndex] / epochs;
					}
					sann.deltaWeights[layerIndex][weightIndex] = 0.0f;
				}
			}
		}
	}
}