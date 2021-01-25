using Newtonsoft.Json;

namespace NeuralNetworksLibrary.Functions
{
    public abstract class ActivationFunction
    {
        [JsonProperty("alpha")]
        protected double alpha;

        public ActivationFunction(double alpha) => this.alpha = alpha;
    }
}
