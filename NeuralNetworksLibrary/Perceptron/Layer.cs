using LinearAlgebraLibrary;

namespace NeuralNetworksLibrary.Perceptron
{
    /// <summary>
    /// Слой перцептрона
    /// </summary>
    internal class Layer
    {
        /// <summary>
        /// Матрица весов на предыдущей итерации обучения
        /// </summary>
        private double[][] lastDeltaWeights;

        /// <summary>
        /// Нейроны этого слоя
        /// </summary>
        private Neuron[] neurons;

        /// <summary>
        /// Конфигурация текущего слоя
        /// </summary>
        public LayerConfig Config { get; }

        /// <summary>
        /// Конструктор слоя
        /// </summary>
        /// <param name="config">Объект конфигурации слоя</param>
        public Layer(LayerConfig config)
        {
            // Сохраняется конфигурация текущего слоя
            Config = config;

            // Задаём размерность матрицы синаптических весов
            lastDeltaWeights = new double[config.NumberOfNeurons][];

            neurons = new Neuron[config.NumberOfNeurons];

            int t = config.NumberOfNeurons + config.NumberOfNeuronsPreviousLayer;

            for (int i = 0; i < config.NumberOfNeurons; i++)
            {

                // Задаём начальные синаптические веса W маленькими случайными значениями
                var W = Vector.GetRandomVector(config.NumberOfNeuronsPreviousLayer + 1, -0.25d / t, 0.25d / t);

                lastDeltaWeights[i] = W.GetCoordinatesSlice(0, config.NumberOfNeuronsPreviousLayer);

                neurons[i] = new Neuron(new Vector(W.GetCoordinatesSlice(1, config.NumberOfNeuronsPreviousLayer)), W[0], config.ActivationFunction);
            }
        }

        /// <summary>
        /// Вычисление состояния слоя
        /// </summary>
        /// <param name="I">Вектор входных данных</param>
        /// <returns>Вектор выхода слоя</returns>
        public Vector Compute(Vector I)
        {
            double[] o = new double[neurons.Length];
            for (int i = 0; i < Config.NumberOfNeurons; i++)
            {
                neurons[i].Inputs = I;
                o[i] = neurons[i].Output;
            }
            return new Vector(o);
        }

        /// <summary>
        /// Вычисления при обратном проходе для скрытого слоя
        /// </summary>
        /// <param name="GrSums"> Градиентные суммы предыдущего слоя </param>
        /// <returns> Вектор градиентных сумм этого слоя </returns>
        public Vector ComputeHiddenBackward(Vector GrSums)
        {
            Vector NewGrSums = Vector.GetZeroVector(Config.NumberOfNeuronsPreviousLayer + 1);
            // подсчёт градиентных сумм скрытого слоя
            for (int j = 0; j < NewGrSums.Size; j++)
            {
                double sum = 0;
                for (int k = 0; k < neurons.Length; k++)
                {
                    if (j == 0)
                    {
                        sum += neurons[k].Bias * neurons[k].Derivative * GrSums[k];
                    }
                    else
                    {
                        sum += neurons[k].Weights[j - 1] * neurons[k].Derivative * GrSums[k];
                    }
                }
                NewGrSums[j] = sum;
            }
            // обновление смещений и весов
            for (int i = 0; i < Config.NumberOfNeurons; ++i)
            {
                double deltabias = Config.Momentum * lastDeltaWeights[i][0] + Config.LearningRate * neurons[i].Derivative * GrSums[i];
                lastDeltaWeights[i][0] = deltabias;
                neurons[i].Bias += deltabias;
                for (int n = 1; n <= Config.NumberOfNeuronsPreviousLayer; ++n)
                {
                    double deltaw = Config.Momentum * lastDeltaWeights[i][n] + Config.LearningRate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * GrSums[i];
                    lastDeltaWeights[i][n] = deltaw;
                    neurons[i].Weights[n - 1] += deltaw;
                }
            }
            return NewGrSums;
        }

        /// <summary>
        /// Вычисления при обратном проходе для выходного слоя
        /// </summary>
        /// <param name="errors"> Вектор ошибок сети </param>
        /// <returns> Вектор градиентных сумм этого слоя </returns>
        public Vector ComputeOutputBackward(Vector errors)
        {
            Vector GrSums = Vector.GetZeroVector(Config.NumberOfNeuronsPreviousLayer + 1);
            //вычисление градиентных сумм выходного слоя
            for (int j = 0; j < GrSums.Size; j++)
            {
                double sum = 0;
                for (int k = 0; k < neurons.Length; ++k)
                {
                    if (j == 0)
                    {
                        sum += neurons[k].Bias * errors[k];
                    }
                    else
                    {

                        sum += neurons[k].Weights[j - 1] * errors[k];
                    }
                }
                GrSums[j] = sum;
            }
            // обновление смещений и весов
            for (int i = 0; i < Config.NumberOfNeurons; ++i)
            {
                double deltabias = Config.Momentum * lastDeltaWeights[i][0] + Config.LearningRate * errors[i];
                lastDeltaWeights[i][0] = deltabias;
                neurons[0].Bias += deltabias;
                for (int n = 1; n <= Config.NumberOfNeuronsPreviousLayer; ++n)
                {
                    double deltaw = Config.Momentum * lastDeltaWeights[i][n] + Config.LearningRate * neurons[i].Inputs[n - 1] * errors[i];
                    lastDeltaWeights[i][n] = deltaw;
                    neurons[i].Weights[n - 1] += deltaw;
                }
            }
            return GrSums;
        }

        public void SetWeights(double[][] weigths)
        {
            for (int i = 0; i < Config.NumberOfNeurons; i++)
            {
                neurons[i].Bias = weigths[i][0];
                neurons[i].Weights = new Vector(new Vector(weigths[i]).GetCoordinatesSlice(1, Config.NumberOfNeuronsPreviousLayer));
            }
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[Config.NumberOfNeurons][];

            for (int i = 0; i < Config.NumberOfNeurons; i++)
            {
                weights[i] = new double[Config.NumberOfNeuronsPreviousLayer + 1];
                weights[i][0] = neurons[i].Bias;
                for (int j = 1; j < weights[i].Length; j++)
                    weights[i][j] = neurons[i].Weights[j - 1];
            }

            return weights;
        }
    }
}
