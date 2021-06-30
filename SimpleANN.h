#pragma once

#include <random>
#include <math.h>
#include <mutex>

// Initialize rand() !

namespace simpleANN
{
	struct ActFunc
	{
	public:
		ActFunc(float (*activation)(float), float (*derivative)(float)) : _activation(activation), _derivative(derivative) {}
		float (*_activation)(float);
		float (*_derivative)(float);
	};
	// Sigmoid function. Outputs value between 0.0f and 1.0f. Slow to calculate so recommended to use for only the output layer.
	static ActFunc sigmoid(
		[](float x) { return 1.0f / (1.0f + expf(-x)); },
		[](float x) { return x * (1.0f - x); }
	);
	// Rectified linear unit function. Outputs 0 when x < 0, and outputs x when x >= 0.
	static ActFunc relu(
		[](float x) { return x < 0.0f ? 0.0f : x; },
		[](float x) { return x < 0.0f ? 0.0f : 1.0f; }
	);
	// Leaky rectified linear unit. Outputs 0.01 * x when x < 0, and outputs x when x >= 0.
	static ActFunc leakyRelu(
		[](float x) { return x < 0.0f ? 0.01f * x : x; },
		[](float x) { return x < 0.0f ? 0.01f : 1.0f; }
	);
	// Hyperbolic tangent function. Outputs tanhf(x) of math.h
	static ActFunc hyperbolicTanh(
		[](float x) { return tanhf(x); },
		[](float x) { return 1.0f - x * x; }
	);


	struct CreateInfo
	{
		ActFunc _hiddenActivationFunction = hyperbolicTanh;
		ActFunc _outputActivationFunction = sigmoid;
		int _inputSize = 1;
		int _outputSize = 1;
		int _hiddenSize = 1;
		int _numberOfHiddenLayers = 1;
		float _learningRate = 0.1f;
		// Momentum between 0.0f and 1.0f. If 0.0f, then no memory is allocated for momentum array.
		// Momentum carries part of the previous delta values over when calculating new weights and biases.
		float _momentum = 0.0f;
	};

	class Layer
	{
	public:
		int _id;
		int _layerSize;
		std::condition_variable _outputInUseCv;
		bool _outputInUse = false;
		Layer* _prevLayer = nullptr;
		Layer* _nextLayer = nullptr;
		float* _outputs = nullptr;
		float* _biases = nullptr;
		float* _weights = nullptr;

		Layer(int id, Layer* previousLayer, int layerSize, float momentum)
			: _prevLayer(previousLayer), _layerSize(layerSize), _momentum(momentum), _id(id)
		{
			_outputs = new float[layerSize];
			for (int i = 0; i < layerSize; ++i)
			{
				_outputs[i] = 0.0f;
			}

			if (previousLayer != nullptr)
			{
				_weights = new float[layerSize * previousLayer->_layerSize];
				_deltaWeights = new float[layerSize * previousLayer->_layerSize];
				_biases = new float[layerSize];
				_deltaBiases = new float[layerSize];
				_error = new float[layerSize];
				if (momentum > 0.0f)
				{
					_weightMomentum = new float[layerSize * previousLayer->_layerSize];
					_biasMomentum = new float[layerSize];
				}
				for (int i = 0; i < layerSize; ++i)
				{
					_biases[i] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
					_deltaBiases[i] = 0.0f;
					if (momentum > 0.0f)
					{
						_biasMomentum[i] = 0.0f;
					}
					_error[i] = 0.0f;
					for (int j = 0; j < previousLayer->_layerSize; ++j)
					{
						_weights[i * _prevLayer->_layerSize + j] = 2.0f * (float(rand()) / float(RAND_MAX)) - 1.0f;
						_deltaWeights[i * _prevLayer->_layerSize + j] = 0.0f;
						if (momentum > 0.0f)
						{
							_weightMomentum[i * _prevLayer->_layerSize + j] = 0.0f;
						}
					}
				}
			}
		}
		~Layer()
		{
			delete[](_outputs);
			delete[](_error);
			delete[](_weights);
			delete[](_deltaWeights);
			delete[](_biases);
			delete[](_deltaBiases);
			if (_momentum > 0)
			{
				delete[](_biasMomentum);
				delete[](_weightMomentum);
			}
		}
		void propagateForward(const float* input)
		{
			std::unique_lock<std::mutex> ul(_lock);
			while (_outputInUse) { _outputInUseCv.wait(ul); }
			_outputInUse = true;
			for (int i = 0; i < _layerSize; ++i)
				_outputs[i] = input[i];
		}
		// Note: output layer stays locked until outputWasRead() is called.
		void propagateForward(float (*actFunc)(float))
		{
			std::unique_lock<std::mutex> ul(_lock);
			while (_outputInUse) { _outputInUseCv.wait(ul); }
			_outputInUse = true;
			for (int i = 0; i < _layerSize; ++i)
			{
				_outputs[i] = 0.0f;
				for (int j = 0; j < _prevLayer->_layerSize; ++j)
					_outputs[i] += _weights[i * _prevLayer->_layerSize + j] * _prevLayer->_outputs[j];
				_outputs[i] += _biases[i];
				_outputs[i] = actFunc(_outputs[i]);
			}
			{
				std::unique_lock<std::mutex> prevUl(_prevLayer->_lock);
				_prevLayer->_outputInUse = false;
			}
			_prevLayer->_outputInUseCv.notify_one();
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
			for (int i = 0; i < _layerSize; ++i)
			{
				_error[i] = label[i] - _outputs[i];
			}
			calculateDerivative(derFunc);
			calculateDelta();
		}
		// Update weights and biases. Update will zero delta values. If the layer has momentum, update will calculate new momentum values.
		void update(float learningRate, int epochs)
		{
			unsigned int node;
			for (int i = 0; i < _layerSize; ++i)
			{
				if (_momentum > 0.0f)
				{
					_biases[i] += learningRate * _deltaBiases[i] / epochs + _biasMomentum[i];
					_biasMomentum[i] = _momentum * _biasMomentum[i] + _momentum * _deltaBiases[i];
				}
				else
				{
					_biases[i] += learningRate * _deltaBiases[i] / epochs;
				}
				_deltaBiases[i] = 0.0f;
				for (int j = 0; j < _prevLayer->_layerSize; ++j)
				{
					node = i * _prevLayer->_layerSize + j;
					if (_momentum > 0.0f)
					{
						_weights[node] += learningRate * _deltaWeights[node] / epochs + _weightMomentum[node];
						_weightMomentum[node] = _momentum * _weightMomentum[node] + _momentum * _deltaWeights[node];
					}
					else
					{
						_weights[node] += learningRate * _deltaWeights[node] / epochs;
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
			for (int i = 0; i < _layerSize; ++i)
			{
				_error[i] = 0.0f;
				for (int j = 0; j < _nextLayer->_layerSize; ++j)
					_error[i] += _nextLayer->_error[j] * _nextLayer->_weights[j * _layerSize + i];
			}
		}
		void calculateDerivative(float (*derFunc)(float))
		{
			for (int i = 0; i < _layerSize; ++i)
				_error[i] = _error[i] * derFunc(_outputs[i]);
		}
		void calculateDelta()
		{
			for (int i = 0; i < _layerSize; ++i)
			{
				_deltaBiases[i] += _error[i];
				for (int j = 0; j < _prevLayer->_layerSize; ++j)
					_deltaWeights[i * _prevLayer->_layerSize + j] += _error[i] * _prevLayer->_outputs[j];
			}
		}
	};

	class ANNetwork
	{
	public:
		Layer* _inputLayer = nullptr;
		Layer* _outputLayer = nullptr;
		ActFunc _hiddenActivationFunction = leakyRelu;
		ActFunc _outputActivationFunction = sigmoid;
		float _learningRate = 0.1f;
		// Momentum between 0.0f and 1.0f. If 0.0f, then no memory is allocated for momentum array.
		// Momentum carries part of the previous delta values over when calculating new weights and biases.

		ANNetwork(const CreateInfo& createInfo) :
			_hiddenActivationFunction(createInfo._hiddenActivationFunction),
			_outputActivationFunction(createInfo._outputActivationFunction)
		{
			_learningRate = createInfo._learningRate;
			_inputLayer = new Layer(0, nullptr, createInfo._inputSize, createInfo._momentum);
			Layer* layer = _inputLayer;
			int lId = 1;
			for (int i = 0; i < createInfo._numberOfHiddenLayers; i++)
			{
				Layer* nextLayer = new Layer(lId++, layer, createInfo._hiddenSize, createInfo._momentum);
				layer->_nextLayer = nextLayer;
				layer = nextLayer;
			}
			_outputLayer = new Layer(lId, layer, createInfo._outputSize, createInfo._momentum);
			layer->_nextLayer = _outputLayer;
		}
		~ANNetwork()
		{
			if (!_outputLayer)
			{
				return;
			}

			Layer* layer = _outputLayer->_prevLayer;
			while (layer->_prevLayer != nullptr)
			{
				delete(layer->_nextLayer);
				layer = layer->_prevLayer;
			}
			delete(layer);
		}
		// Propagate the network forward. After calling propagateForward(), output array of the last layer will contain the networks output.
		void propagateForward(const float* input)
		{
			_inputLayer->propagateForward(input);
			Layer* layer = _inputLayer->_nextLayer;
			while (layer->_nextLayer != nullptr)
			{
				layer->propagateForward(_hiddenActivationFunction._activation);
				layer = layer->_nextLayer;
			}
			layer->propagateForward(_outputActivationFunction._activation);
		}
		// Propagate backward. Error of the output layer will be (target - output).
		void propagateBackward(const float* labels)
		{
			Layer* layer = _outputLayer;
			layer->propagateBackward(labels, _outputActivationFunction._derivative);
			layer = layer->_prevLayer;

			while (layer->_prevLayer != nullptr)
			{
				layer->propagateBackward(_hiddenActivationFunction._derivative);
				layer = layer->_prevLayer;
			}
		}
		void update(int batchSize)
		{
			Layer* layer = _outputLayer;
			while (layer->_prevLayer != nullptr)
			{
				layer->update(_learningRate, batchSize);
				layer = layer->_prevLayer;
			}
		}
		void outputWasRead()
		{
			_outputLayer->_outputInUse = false;
			_outputLayer->_outputInUseCv.notify_one();
		}
	};
}