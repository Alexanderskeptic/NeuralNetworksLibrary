using Newtonsoft.Json;
using System;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Текучий выпрямитель
    /// </summary>
    public class ELU : ActivationFunction, IActivationFunction
    {
        public ELU(double alpha) : base(alpha) { }

        public double Activate(double x) => x < 0 ? alpha * (Math.Exp(x) - 1.0) : x;

        public double Derivate(double x) => alpha * Math.Exp(x) < 0 ? alpha : 1.0;
    }
}
