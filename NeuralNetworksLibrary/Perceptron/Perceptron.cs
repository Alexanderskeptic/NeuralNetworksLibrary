using Newtonsoft.Json;
using System.Diagnostics;
using System.IO;
using LinearAlgebraLibrary;
using System;

namespace NeuralNetworksLibrary.Perceptron
{
    /// <summary>
    /// Многослойная нейронная сеть прямого распространения, называемая также многослойным персептроном
    /// </summary>
    public class Perceptron
    {
        /// <summary>
        /// Слои перцептрона
        /// </summary>
        Layer[] layers;

        /// <summary>
        /// Индекс выходного слоя
        /// </summary>
        int outputLayerIndex;

        /// <summary>
        /// Значение фукнции стоимости (ошибки), преодолённое за последнее обучение
        /// </summary>
        public double LastError { get; private set; }

        /// <summary>
        /// Длительность последнего обучения
        /// </summary>
        public TimeSpan LastLearningTime { get; private set; }

        /// <summary>
        /// Конструктор.
        /// На вход следует подавать данные без учёта входного слоя, 
        /// так как он не является вычислительным
        /// </summary>
        /// <param name="configs">Конфигурация слоёв</param>
        public Perceptron(params LayerConfig[] configs)
        {
            if (configs != null)
            {
                layers = new Layer[configs.Length];

                // Номер выходного слоя (последний слой)
                outputLayerIndex = layers.Length - 1;

                // Создаются слои нейронной сети
                for (int i = 0; i <= outputLayerIndex; i++)
                    layers[i] = new Layer(configs[i]);
            }
        }

        /// <summary>
        /// Получить предсказание перцептрона
        /// </summary>
        /// <param name="I">Вектор входных данных</param>
        /// <returns>Некий вектор выходных данных</returns>
        public Vector Predict(Vector I)
        {
            Vector Oj = layers[0].Compute(I);
            for (int j = 1; j < layers.Length; j++)
            {
                Oj = layers[j].Compute(Oj);
            }
            return Oj;
        }

        /// <summary>
        /// Ошибка одной итерации обучения
        /// </summary>
        /// <param name="errors">Вектор ошибок итерации обучения</param>
        /// <returns></returns>
        double GetMSE(Vector errors) => (errors * errors) * 0.5d;

        /// <summary>
        /// Ошибка эпохи
        /// </summary>
        /// <param name="mses">Вектор ошибок итераций</param>
        /// <returns></returns>
        double GetCost(Vector mses) => (mses * Vector.GetVectorOfOnes(mses.Size)) / mses.Size;

        /// <summary>
        /// Обучение перцептрона методом обратного распространения ошибки.
        /// Алгоритм обратного распространения ошибки является одним из методов обучения многослойных нейронных сетей прямого распространения.
        /// 
        /// Обучение алгоритмом обратного распространения ошибки предполагает два прохода по всем слоям сети: прямого и обратного.
        /// Прямой проход.
        /// При прямом проходе входной вектор подается на входной слой нейронной сети, после чего распространяется по сети от слоя к слою. 
        /// В результате генерируется набор выходных сигналов, который и является фактической реакцией сети на данный входной образ. 
        /// Во время прямого прохода все синаптические веса сети фиксированы.
        /// Обратный проход.
        /// Во время обратного прохода все синаптические веса настраиваются в соответствии с правилом коррекции ошибок, а именно: 
        /// фактический выход сети вычитается из желаемого, в результате чего формируется сигнал ошибки. 
        /// Этот сигнал впоследствии распространяется по сети в направлении, обратном направлению синаптических связей. Отсюда и название – алгоритм обратного распространения ошибки. 
        /// Синаптические веса настраиваются с целью максимального приближения выходного сигнала сети к желаемому.
        /// </summary>
        /// <param name="trainset"> Обучающая выборка </param>
        /// <param name="threshold"> Порог функции стоимости, по умолчанию 0.001 </param>
        public void TrainBackProp((Vector X, Vector Y)[] trainset, double threshold = 0.001d)
        {
            // Засекаем время обучения нейронной сети
            var w = Stopwatch.StartNew();
            Vector MSEs = Vector.GetZeroVector(trainset.Length);
            double epochCost = 0;
            int epoch = 0;
            do
            {
                for (int i = 0; i < trainset.Length; i++)
                {
                    Vector O = Predict(trainset[i].X);
                    Vector E = trainset[i].Y - O;
                    MSEs[i] = GetMSE(E);
                    Vector GSums = layers[outputLayerIndex].ComputeOutputBackward(E);
                    for (int j = outputLayerIndex - 1; j >= 0; j--)
                    {
                        GSums = layers[j].ComputeHiddenBackward(GSums);
                    }
                }
                epoch++;
                epochCost = GetCost(MSEs);
#if DEBUG
                WriteLine($"{epochCost} - epoch {epoch}");
#endif
            } while (epochCost > threshold);

            // Обучение окончено
            w.Stop();

            LastError = epochCost;
            LastLearningTime = TimeSpan.FromMilliseconds(w.ElapsedMilliseconds);
#if DEBUG
            WriteLine($"{w.ElapsedMilliseconds / 1000d} seconds");
#endif
        }

        #region Сохранить модель в JSON файл
        /// <summary>
        /// Сохранить модель в JSON файл
        /// </summary>
        /// <param name="path"> Путь сохранения, включая название файла и формат </param>
        public void SaveModel(string FileName)
        {
            if (string.IsNullOrWhiteSpace(FileName))
                throw new ArgumentException("File name is empty", nameof(FileName));

            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                TypeNameHandling = TypeNameHandling.Auto
            };

            var model = new
            {
                lastError = LastError,
                lastLearningTime = LastLearningTime,
                parameters = new object[layers.Length]
            };

            for (int i = 0; i < model.parameters.Length; i++)
            {
                model.parameters[i] = new
                {
                    config = layers[i].Config,
                    weights = layers[i].GetWeights()
                };
            }

            using (StreamWriter file = File.CreateText(FileName))
            {
                JsonSerializer serializer = new JsonSerializer
                {
                    Formatting = Formatting.Indented,
                    TypeNameHandling = TypeNameHandling.Auto
                };
                serializer.Serialize(file, model);
            }
        }
        #endregion

        #region Загрузить модель из JSON файла
        /// <summary>
        /// Загрузить модель из JSON файла
        /// </summary>
        /// <param name="path">Путь к файлу</param>
        public void LoadModel(string FileName = null)
        {
            if (string.IsNullOrWhiteSpace(FileName))
                throw new ArgumentException("File name is empty", nameof(FileName));

            if (!File.Exists(FileName))
                throw new ArgumentException("file does not exist", FileName);

            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                TypeNameHandling = TypeNameHandling.Auto
            };
            var model = new
            {
                lastError = 1.01d,
                lastTime = TimeSpan.FromSeconds(1.25d),
                parameters = new[]
                {
                    new
                    {
                        config = new LayerConfig(),
                        weights = new double[1][]
                    }
                }
            };
            using (StreamReader r = new StreamReader(FileName))
            {
                string json = r.ReadToEnd();
                model = JsonConvert.DeserializeAnonymousType(json, model, settings);
            }
            if (model != null)
            {
                layers = new Layer[model.parameters.Length];
                outputLayerIndex = layers.Length - 1;
                for (int i = 0; i < layers.Length; i++)
                {
                    LastError = model.lastError;
                    LastLearningTime = model.lastTime;
                    layers[i] = new Layer(model.parameters[i].config);
                    layers[i].SetWeights(model.parameters[i].weights);
                }
            }
        }
        #endregion

        
    }
}
