using Newtonsoft.Json;
using static System.Math;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Сигмовидная функция
    /// </summary>
    public class Sigmoid : ActivationFunction, IActivationFunction
    {
        public Sigmoid(double alpha) : base(alpha) { }

        public double Activate(double x) => 1.0 / (1.0 + Exp(alpha * (-x)));

        public double Derivate(double x) => (alpha * Exp(alpha * (-x))) / Pow((Exp(alpha * (-x)) + 1.0), 2);
    }
}
