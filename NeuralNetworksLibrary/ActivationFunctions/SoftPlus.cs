using Newtonsoft.Json;
using static System.Math;

namespace NeuralNetworksLibrary.ActivationFunctions
{
    /// <summary>
    /// Смягчённый выпрямитель
    /// </summary>
    public class SoftPlus : ActivationFunction, IActivationFunction
    {
        public SoftPlus(double alpha) : base(alpha) { }

        public double Activate(double x) => Log(1.0 + Exp(alpha * x));

        public double Derivate(double x) => (alpha * Exp(alpha * x)) / (1.0 + Exp(alpha * x));
    }
}
