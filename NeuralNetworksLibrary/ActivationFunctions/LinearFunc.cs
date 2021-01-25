using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksLibrary.Functions
{
    /// <summary>
    /// Линейная функция
    /// </summary>
    public class LinearFunc : ActivationFunction, IActivationFunction
    {
        [JsonProperty("beta")]
        private double beta;

        public LinearFunc(double alpha, double beta) : base(alpha) { }

        public double Activate(double x) => alpha * x + beta;

        public double Derivate(double x)
        {
            return alpha;
        }
    }
}
