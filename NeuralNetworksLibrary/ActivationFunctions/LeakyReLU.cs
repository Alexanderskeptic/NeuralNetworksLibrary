﻿using Newtonsoft.Json;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Текучий выпрямитель
    /// </summary>
    public class LeakyReLU : ActivationFunction, IActivationFunction
    {
        public LeakyReLU(double alpha) : base(alpha) { }

        public double Activate(double x) => x < 0 ? alpha * x : x;

        public double Derivate(double x) => x < 0 ? alpha : 1.0;
    }
}
