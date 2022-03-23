#pragma once

#include <random>
#include <math.h>
#include <mutex>

// Initialize rand() !

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

	struct CreateInfo
	{
		float (*hiddenActFunc)(float) = leakyReluAct;
		float (*hiddenDerFunc)(float) = leakyReluDer;
		float (*outActFunc)(float) = sigmoidAct;
		float (*outDerFunc)(float) = sigmoidDer;
		int inputSize = 1;
		int outputSize = 1;
		int hiddenSize = 1;
		int numberOfHiddenLayers = 1;
		float learningRate = 0.1f;
		// Momentum between 0.0f and 1.0f. If 0.0f, then no memory is allocated for momentum array.
		// Momentum carries part of the previous delta values over when calculating new weights and biases.
		float momentum = 0.0f;
	};

	class Layer
	{
	public:
		int layerSize;
		std::condition_variable outputInUseCv;
		bool outputInUse = false;
		Layer* prevLayer = nullptr;
		Layer* nextLayer = nullptr;
		float* outputs = nullptr;
		float* biases = nullptr;
		float* weights = nullptr;

		Layer(Layer* previousLayer, int layerSize, float momentum) : prevLayer(previousLayer), layerSize(layerSize), _momentum(momentum)
		{
			outputs = new float[layerSize];
			for (int i = 0; i < layerSize; ++i)
				outputs[i] = 0.0f;
			
			if (previousLayer == nullptr)
				return;

			weights = new float[layerSize * previousLayer->layerSize];
			biases = new float[layerSize];
			_deltaWeights = new float[layerSize * previousLayer->layerSize] { 0.0f };
			_deltaBiases = new float[layerSize] { 0.0f };
			_error = new float[layerSize] { 0.0f };
			for (int i = 0; i < layerSize; ++i)
			{
				biases[i] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
				for (int j = 0; j < previousLayer->layerSize; ++j)
				{
					weights[i * prevLayer->layerSize + j] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
				}
			}
			if (momentum != 0.0f)
			{
				_weightMomentum = new float[layerSize * previousLayer->layerSize] { 0.0f };
				_biasMomentum = new float[layerSize] { 0.0f };
			}
		}
		~Layer()
		{
			delete[](outputs);
			delete[](_error);
			delete[](weights);
			delete[](_deltaWeights);
			delete[](biases);
			delete[](_deltaBiases);
			if (_momentum > 0)
			{
				delete[](_biasMomentum);
				delete[](_weightMomentum);
			}
		}
		Layer(const Layer& other) = delete;
		Layer(Layer&& other) = delete;
		Layer& operator=(const Layer& other) = delete;
		Layer& operator=(Layer&& other) = delete;
		
		void propagateForward(const float* input)
		{
			std::unique_lock<std::mutex> ul(_lock);
			while (outputInUse) { outputInUseCv.wait(ul); }
			outputInUse = true;
			for (int i = 0; i < layerSize; ++i)
				outputs[i] = input[i];
		}
		// Note: output layer stays locked until outputWasRead() is called.
		void propagateForward(float (*actFunc)(float))
		{
			std::unique_lock<std::mutex> ul(_lock);
			while (outputInUse) { outputInUseCv.wait(ul); }
			outputInUse = true;
			for (int i = 0; i < layerSize; ++i)
			{
				outputs[i] = 0.0f;
				for (int j = 0; j < prevLayer->layerSize; ++j)
					outputs[i] += weights[i * prevLayer->layerSize + j] * prevLayer->outputs[j];
				outputs[i] += biases[i];
				outputs[i] = actFunc(outputs[i]);
			}
			{
				std::unique_lock<std::mutex> prevUl(prevLayer->_lock);
				prevLayer->outputInUse = false;
			}
			prevLayer->outputInUseCv.notify_one();
		}

		// Propagate backward. Delta weights and biases are cumulatively added on each back propagation.
		void propagateBackward(float (*derFunc)(float))
		{
			calculateError();
			calculateDerivative(derFunc);
			calculateDelta();
		}
		// Propagate backward with error of (target - output).
		void propagateBackward(const float* label, float (*derFunc)(float))
		{
			for (int i = 0; i < layerSize; ++i)
				_error[i] = label[i] - outputs[i];
			calculateDerivative(derFunc);
			calculateDelta();
		}
		// Update weights and biases. Update will zero delta values. If the layer has momentum, update will calculate new momentum values.
		void update(float learningRate, int epochs)
		{
			unsigned int node;
			for (int i = 0; i < layerSize; ++i)
			{
				if (_momentum > 0.0f)
				{
					biases[i] += learningRate * _deltaBiases[i] / epochs + _biasMomentum[i];
					_biasMomentum[i] = _momentum * _biasMomentum[i] + _momentum * _deltaBiases[i];
				}
				else
				{
					biases[i] += learningRate * _deltaBiases[i] / epochs;
				}
				_deltaBiases[i] = 0.0f;
				for (int j = 0; j < prevLayer->layerSize; ++j)
				{
					node = i * prevLayer->layerSize + j;
					if (_momentum > 0.0f)
					{
						weights[node] += learningRate * _deltaWeights[node] / epochs + _weightMomentum[node];
						_weightMomentum[node] = _momentum * _weightMomentum[node] + _momentum * _deltaWeights[node];
					}
					else
					{
						weights[node] += learningRate * _deltaWeights[node] / epochs;
					}
					_deltaWeights[node] = 0.0f;
				}
			}
		}

	private:
		std::mutex _lock;
		float* _deltaWeights = nullptr;
		float* _deltaBiases = nullptr;
		float* _weightMomentum = nullptr;
		float* _biasMomentum = nullptr;
		float* _error = nullptr;
		float _momentum = 0.0f; // Momentum value. If momentum is 0.0f, no memory is allocated for momentum.

		void calculateError()
		{
			for (int i = 0; i < layerSize; ++i)
			{
				_error[i] = 0.0f;
				for (int j = 0; j < nextLayer->layerSize; ++j)
					_error[i] += nextLayer->_error[j] * nextLayer->weights[j * layerSize + i];
			}
		}
		void calculateDerivative(float (*derFunc)(float))
		{
			for (int i = 0; i < layerSize; ++i)
				_error[i] = _error[i] * derFunc(outputs[i]);
		}
		void calculateDelta()
		{
			for (int i = 0; i < layerSize; ++i)
			{
				_deltaBiases[i] += _error[i];
				for (int j = 0; j < prevLayer->layerSize; ++j)
					_deltaWeights[i * prevLayer->layerSize + j] += _error[i] * prevLayer->outputs[j];
			}
		}
	};

	class ANNetwork
	{
	public:
		Layer* inputLayer = nullptr;
		Layer* outputLayer = nullptr;

		ANNetwork(const CreateInfo& createInfo)
		{
			_hiddenActFunc = createInfo.hiddenActFunc;
			_hiddenDerFunc = createInfo.hiddenDerFunc;
			_outActFunc = createInfo.outActFunc;
			_outDerFunc = createInfo.outDerFunc;
			_learningRate = createInfo.learningRate;
			inputLayer = new Layer(0, nullptr, createInfo.inputSize, createInfo.momentum);
			Layer* layer = inputLayer;
			int lId = 1;
			for (int i = 0; i < createInfo.numberOfHiddenLayers; i++)
			{
				Layer* nextLayer = new Layer(lId++, layer, createInfo.hiddenSize, createInfo.momentum);
				layer->nextLayer = nextLayer;
				layer = nextLayer;
			}
			outputLayer = new Layer(lId, layer, createInfo.outputSize, createInfo.momentum);
			layer->nextLayer = outputLayer;
		}
		~ANNetwork()
		{
			if (!outputLayer)
				return;
			Layer* layer = outputLayer->prevLayer;
			while (layer->prevLayer != nullptr)
			{
				delete(layer->nextLayer);
				layer = layer->prevLayer;
			}
			delete(layer);
		}
		ANNetwork(const ANNetwork& other) = delete;
		ANNetwork(ANNetwork&& other) = delete;
		ANNetwork& operator=(const ANNetwork& other) = delete;
		ANNetwork& operator=(ANNetwork&& other) = delete;
		
		// Propagate the network forward. After calling propagateForward(), output array of the last layer will contain the networks output.
		void propagateForward(const float* input)
		{
			inputLayer->propagateForward(input);
			Layer* layer = inputLayer->nextLayer;
			while (layer->nextLayer != nullptr)
			{
				layer->propagateForward(_hiddenActFunc);
				layer = layer->nextLayer;
			}
			layer->propagateForward(_outActFunc);
		}
		// Propagate backward. Error of the output layer will be (target - output).
		void propagateBackward(const float* labels)
		{
			Layer* layer = outputLayer;
			layer->propagateBackward(labels, _outDerFunc);
			layer = layer->prevLayer;

			while (layer->prevLayer != nullptr)
			{
				layer->propagateBackward(_hiddenDerFunc);
				layer = layer->prevLayer;
			}
		}
		void update(int batchSize)
		{
			Layer* layer = outputLayer;
			while (layer->prevLayer != nullptr)
			{
				layer->update(_learningRate, batchSize);
				layer = layer->prevLayer;
			}
		}
		void outputWasRead()
		{
			outputLayer->outputInUse = false;
			outputLayer->outputInUseCv.notify_one();
		}
		
	private:
		float (*_hiddenActFunc)(float);
		float (*_hiddenDerFunc)(float);
		float (*_outActFunc)(float);
		float (*_outDerFunc)(float);
		float _learningRate = 0.1f;
		// Momentum between 0.0f and 1.0f. If 0.0f, then no memory is allocated for momentum array.
		// Momentum carries part of the previous delta values over when calculating new weights and biases.
	};
}