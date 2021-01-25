using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworksLibrary.ActivationFunctions;

namespace NeuralNetworksLibraryUnitTests
{
    [TestClass]
    public class ActivationFunctionsUnitTests
    {
        #region Arctan tests
        [TestMethod]
        public void ArctanTestMethod1()
        {
            // arrange
            double[] x = new double[5] { -2.5, -1, 0, 1, 2.5 };
            double alpha = 1;
            double beta = 1;
            Arctan arctan = new Arctan(alpha, beta);
            double[] resultFunctionValue = new double[5];
            double[] resultFunctionDerivative = new double[5];
            double[] ExpectedFunctionValue = new double[5] { -1.1902899496825317, -0.7853981633974483, 0, 0.7853981633974483, 1.1902899496825317 };
            double[] ExpectedFunctionDerivative = new double[5] { 0.13793103448275862, 0.5, 1, 0.5, 0.13793103448275862 };

            // act
            for (int i = 0; i < 5; i++)
            {
                resultFunctionValue[i] = arctan.Activate(x[i]);
                resultFunctionDerivative[i] = arctan.Derivate(x[i]);
            }

            // assert
            Assert.AreEqual(ExpectedFunctionValue[0], resultFunctionValue[0]);
            Assert.AreEqual(ExpectedFunctionValue[1], resultFunctionValue[1]);
            Assert.AreEqual(ExpectedFunctionValue[2], resultFunctionValue[2]);
            Assert.AreEqual(ExpectedFunctionValue[3], resultFunctionValue[3]);
            Assert.AreEqual(ExpectedFunctionValue[4], resultFunctionValue[4]);

            Assert.AreEqual(ExpectedFunctionDerivative[0], resultFunctionDerivative[0]);
            Assert.AreEqual(ExpectedFunctionDerivative[1], resultFunctionDerivative[1]);
            Assert.AreEqual(ExpectedFunctionDerivative[2], resultFunctionDerivative[2]);
            Assert.AreEqual(ExpectedFunctionDerivative[3], resultFunctionDerivative[3]);
            Assert.AreEqual(ExpectedFunctionDerivative[4], resultFunctionDerivative[4]);
        }

        [TestMethod]
        public void ArctanTestMethod2()
        {
            // arrange
            double[] x = new double[5] { -2.5, -1, 0, 1, 2.5 };
            double alpha = -2.3;
            double beta = 0.7;
            Arctan arctan = new Arctan(alpha, beta);
            double[] resultFunctionValue = new double[5];
            double[] resultFunctionDerivative = new double[5];
            double[] ExpectedFunctionValue = new double[5] { 2.4187954888612593, 1.4046697180951795, 0, -1.4046697180951795, -2.4187954888612593 };
            double[] ExpectedFunctionDerivative = new double[5] { -0.39630769230769225, -1.0805369127516777, -1.6099999999999999, -1.0805369127516777, -0.39630769230769225 };

            // act
            for (int i = 0; i < 5; i++)
            {
                resultFunctionValue[i] = arctan.Activate(x[i]);
                resultFunctionDerivative[i] = arctan.Derivate(x[i]);
            }

            // assert
            Assert.AreEqual(ExpectedFunctionValue[0], resultFunctionValue[0]);
            Assert.AreEqual(ExpectedFunctionValue[1], resultFunctionValue[1]);
            Assert.AreEqual(ExpectedFunctionValue[2], resultFunctionValue[2]);
            Assert.AreEqual(ExpectedFunctionValue[3], resultFunctionValue[3]);
            Assert.AreEqual(ExpectedFunctionValue[4], resultFunctionValue[4]);

            Assert.AreEqual(ExpectedFunctionDerivative[0], resultFunctionDerivative[0]);
            Assert.AreEqual(ExpectedFunctionDerivative[1], resultFunctionDerivative[1]);
            Assert.AreEqual(ExpectedFunctionDerivative[2], resultFunctionDerivative[2]);
            Assert.AreEqual(ExpectedFunctionDerivative[3], resultFunctionDerivative[3]);
            Assert.AreEqual(ExpectedFunctionDerivative[4], resultFunctionDerivative[4]);
        }
        #endregion

        #region ELU tests
        [TestMethod]
        public void ELUTestMethod1()
        {
            // arrange
            double[] x = new double[5] { -2.5, -1, 0, 1, 2.5 };
            double alpha = 1;
            ELU eLU = new ELU(alpha);
            double[] resultFunctionValue = new double[5];
            double[] resultFunctionDerivative = new double[5];
            double[] ExpectedFunctionValue = new double[5] { -0.9179150013761012, -0.6321205588285577, 0, 1, 2.5 };
            double[] ExpectedFunctionDerivative = new double[5] { 0.08208499862389884, 0.36787944117144233, 1, 1, 1 };

            // act
            for (int i = 0; i < 5; i++)
            {
                resultFunctionValue[i] = eLU.Activate(x[i]);
                resultFunctionDerivative[i] = eLU.Derivate(x[i]);
            }

            // assert
            Assert.AreEqual(ExpectedFunctionValue[0], resultFunctionValue[0]);
            Assert.AreEqual(ExpectedFunctionValue[1], resultFunctionValue[1]);
            Assert.AreEqual(ExpectedFunctionValue[2], resultFunctionValue[2]);
            Assert.AreEqual(ExpectedFunctionValue[3], resultFunctionValue[3]);
            Assert.AreEqual(ExpectedFunctionValue[4], resultFunctionValue[4]);

            Assert.AreEqual(ExpectedFunctionDerivative[0], resultFunctionDerivative[0]);
            Assert.AreEqual(ExpectedFunctionDerivative[1], resultFunctionDerivative[1]);
            Assert.AreEqual(ExpectedFunctionDerivative[2], resultFunctionDerivative[2]);
            Assert.AreEqual(ExpectedFunctionDerivative[3], resultFunctionDerivative[3]);
            Assert.AreEqual(ExpectedFunctionDerivative[4], resultFunctionDerivative[4]);
        }

        [TestMethod]
        public void ELUTestMethod2()
        {
            // arrange
            double[] x = new double[5] { -2.5, -1, 0, 1, 2.5 };
            double alpha = 2.3;
            ELU eLU = new ELU(alpha);
            double[] resultFunctionValue = new double[5];
            double[] resultFunctionDerivative = new double[5];
            double[] ExpectedFunctionValue = new double[5] { -2.1112045031650326, -1.4538772853056825, 0, 1, 2.5 };
            double[] ExpectedFunctionDerivative = new double[5] { 0.1887954968349672, 0.8461227146943173, 2.3, 1, 1 };

            // act
            for (int i = 0; i < 5; i++)
            {
                resultFunctionValue[i] = eLU.Activate(x[i]);
                resultFunctionDerivative[i] = eLU.Derivate(x[i]);
            }

            // assert
            Assert.AreEqual(ExpectedFunctionValue[0], resultFunctionValue[0]);
            Assert.AreEqual(ExpectedFunctionValue[1], resultFunctionValue[1]);
            Assert.AreEqual(ExpectedFunctionValue[2], resultFunctionValue[2]);
            Assert.AreEqual(ExpectedFunctionValue[3], resultFunctionValue[3]);
            Assert.AreEqual(ExpectedFunctionValue[4], resultFunctionValue[4]);

            Assert.AreEqual(ExpectedFunctionDerivative[0], resultFunctionDerivative[0]);
            Assert.AreEqual(ExpectedFunctionDerivative[1], resultFunctionDerivative[1]);
            Assert.AreEqual(ExpectedFunctionDerivative[2], resultFunctionDerivative[2]);
            Assert.AreEqual(ExpectedFunctionDerivative[3], resultFunctionDerivative[3]);
            Assert.AreEqual(ExpectedFunctionDerivative[4], resultFunctionDerivative[4]);
        }
        #endregion
    }
}
