using Newtonsoft.Json;
using System;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Exponential Linear Units (ELU)
    /// </summary>
    public class ELU : ActivationFunction, IActivationFunction
    {
        public ELU(double alpha) : base(alpha) { }

        public double Activate(double x) => (x > 0)? x : alpha * (Math.Exp(x) - 1.0);

        public double Derivate(double x) => (x > 0)? 1.0 : alpha + Activate(x);
    }
}
