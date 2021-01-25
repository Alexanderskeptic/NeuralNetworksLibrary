using Newtonsoft.Json;
using System;
using static System.Math;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Арктангенс
    /// </summary>
    public class Arctan : ActivationFunction, IActivationFunction
    {
        [JsonProperty("beta")]
        private double beta;

        public Arctan(double alpha, double beta) : base(alpha) => this.beta = beta;

        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Activate(double x) => alpha * Atan(beta * x);

        /// <summary>
        /// Первая производная функции активации
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Derivate(double x) => (alpha * beta) / (Pow(beta, 2) * Pow(x, 2) + 1);
    }
}
