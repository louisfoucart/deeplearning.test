package iris

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

/**
 * Created by louis on 29/04/2016.
 */

class LoggerAnchor

val log = LoggerFactory.getLogger(LoggerAnchor::class.java)

fun main(args : Array<String>) {
    // Customizing params
    Nd4j.MAX_SLICES_TO_PRINT = -1
    Nd4j.MAX_ELEMENTS_PER_SLICE = -1
    val numRows = 4
    val numColumns = 1
    val outputNum = 3
    val numSamples = 150
    val batchSize = 150
    val iterations = 5
    val splitTrainNum = (batchSize * .8).toInt()
    val seed = 123
    val listenerFreq = 1

    log.info("Load data....")
    val iter = IrisDataSetIterator(batchSize, numSamples)
    val next = iter.next()
    next.shuffle()
    next.normalizeZeroMeanZeroUnitVariance()

    log.info("Split data....")
    val testAndTrain = next.splitTestAndTrain(splitTrainNum)
    val train = testAndTrain.train
    val test = testAndTrain.test
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    log.info("Build model....")
    val conf = NeuralNetConfiguration.Builder()
            .seed(seed) // Locks in weight initialization for tuning
            .iterations(iterations) // # training iterations predict/classify & backprop
            .learningRate(1e-6) // Optimization step size
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients
            .l1(1e-1)
            .regularization(true)
            .l2(2e-4)
            .useDropConnect(true)
            .list(
                    RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                    .nIn(numRows * numColumns) // # input nodes
                    .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                    .weightInit(WeightInit.XAVIER) // Weight initialization
                    .k(1) // # contrastive divergence iterations
                    .activation("relu") // Activation function type
                    .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                    .updater(Updater.ADAGRAD)
                    .dropOut(0.5)
                    .build(),
                    OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nIn(3) // # input nodes
                    .nOut(outputNum) // # output nodes
                    .activation("softmax")
                    .build()
            )
            .build()
    val model = MultiLayerNetwork(conf)
    model.init()

    model.setListeners(ScoreIterationListener(listenerFreq))
    log.info("Train model....")
    model.fit(train)

    log.info("Evaluate weights....")
    for (layer in model.layers) {
        val w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
        log.info("Weights: " + w)
    }

    log.info("Evaluate model....")
    val eval = Evaluation(outputNum)
    eval.eval(test.labels, model.output(test.featureMatrix, Layer.TrainingMode.TEST))
    log.info(eval.stats())

    log.info("****************Example finished********************")
}