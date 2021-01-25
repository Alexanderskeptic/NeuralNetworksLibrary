using Newtonsoft.Json;
using System;
using static System.Math;

namespace NeuralNetworksLibrary.Functions
{
    /// <summary>
    /// Гиперболический тангенс
    /// </summary>
    public class Tanh : ActivationFunction, IActivationFunction
    {
        [JsonProperty("beta")]
        private double beta;

        public Tanh(double alpha, double beta) : base(alpha) => this.beta = beta;

        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Activate(double x) => alpha * Math.Tanh(beta * x);

        /// <summary>
        /// Первая производная функции активации
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derivate(double x) => alpha * beta * (1.0 - Math.Pow(Tanh(beta * x), 2));
    }
}
