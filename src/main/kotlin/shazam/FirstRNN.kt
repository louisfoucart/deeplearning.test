package shazam

import org.canova.api.records.reader.SequenceRecordReader
import org.canova.api.records.reader.impl.CSVSequenceRecordReader
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import util.NumberedAdhocFileInputSplit
import java.io.File
import java.util.*

fun main(args: Array<String>) {
    val trainDataPath = "data_train"
    val inputWidth = 13 // number of bands
    val codeMap: Map<String, Int> = readCodeMap(trainDataPath, inputWidth)
    val numberOfLabels = codeMap.size
    val numberOfExamples = 100 // 1042
    val miniBatchSize = 50 // 100
    val hiddenLayerWidth = 10
    val hiddenLayerNumber = 1

    log.info("create network")
    val conf = buildMultiLayerConfiguration(inputWidth, numberOfLabels, hiddenLayerNumber, hiddenLayerWidth)
    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners(ScoreIterationListener(1))

    log.info("total number of network parameters: {}", getTotalNumberOfParameters(net));

    log.info("read train dataset")
    val datasetIterator = readDataset(trainDataPath, inputWidth, "train", numberOfExamples, numberOfLabels, miniBatchSize)

    log.info("split train and test")
    val seed: Long = 123
    val testInput: MutableList<DataSet> = ArrayList()
    while(datasetIterator.hasNext()) {
        val dataSet = datasetIterator.next()
        log.info("test data number {}", dataSet.numExamples())
        val trainTest = dataSet.splitTestAndTrain((dataSet.numExamples()*.8).toInt(), Random(seed))
        testInput.add(trainTest.test)
        net.fit(trainTest.train)
    }
    //net.rnnClearPreviousState() // TODO check to see if it goes into train fit loop

    log.info("evaluate model...")
    val evaluation = Evaluation()
    testInput.forEach {
        val features = it.getFeatureMatrix()
        val labels = it.getLabels()
        val inMask = it.getFeaturesMaskArray()
        val outMask = it.getLabelsMaskArray()
        val predicted = net.output(features, false, inMask, outMask)
        evaluation.evalTimeSeries(labels, predicted, outMask)
    }
    log.info(evaluation.stats())

    log.info("**************** finished ********************")

    // some epochs
//    val epochs = 1 //100
//    for (epoch in 0..epochs-1) {
//        System.out.println("Epoch " + epoch)

    //System.out.println("train network using data")
    //net.fit(datasetIterator)

    //System.out.println("clear current stance from the last example")
    //net.rnnClearPreviousState()

//    val evaluation = Evaluation(labelMap)

//    val testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10)
//    while(testData.hasNext()) {
//        val dsTest = testData.next()
//        val predicted = net.output(dsTest.getFeatureMatrix(), false)
//        val actual = dsTest.getLabels()
//        evaluation.evalTimeSeries(predicted, actual)
//    }

//    }
}

fun getTotalNumberOfParameters(net: MultiLayerNetwork): Int {
    //Print the  number of parameters in the network for each layer
    var totalNumParams = 0
    var index = 0
    net.getLayers().forEach {
        val nParams = it.numParams()
        log.info("number of parameters in layer {}: {}", index++, nParams)
        totalNumParams += nParams
    }
    return totalNumParams
}

fun readCodeMap(inputPath: String, inputWidth: Int): Map<String, Int> {
    val codeMap: MutableMap<String, Int> = HashMap()
    File("$inputPath/nc${inputWidth}_labelCodeMap.csv").bufferedReader().forEachLine {
        val (key, value) = it.split("=")
        codeMap.put(key, value.toInt())
    }
    return codeMap
}

fun buildMultiLayerConfiguration(inputWidth: Int, numberOfLabels: Int, hiddenLayerCount: Int, hiddenLayerWidth: Int): MultiLayerConfiguration {
    // some common parameters
    val builder = NeuralNetConfiguration.Builder()
    builder.iterations(10)
    builder.learningRate(0.001)
    builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    builder.seed(123)
    builder.biasInit(0.0)
    builder.miniBatch(false)
    builder.updater(Updater.RMSPROP)
    builder.weightInit(WeightInit.XAVIER)

    val listBuilder = builder.list(hiddenLayerCount + 1 + 1)

    val inputLayerBuilder = GravesLSTM.Builder()
    inputLayerBuilder.nIn(inputWidth)
    inputLayerBuilder.nOut(hiddenLayerWidth)
    // adopted activation function from GravesLSTMCharModellingExample
    // seems to work well with RNNs
    inputLayerBuilder.activation("tanh")
    listBuilder.layer(0, inputLayerBuilder.build())

    // first difference, for rnns we need to use GravesLSTM.Builder
    for (i in 1..hiddenLayerCount) {
        val hiddenLayerBuilder = GravesLSTM.Builder()
        hiddenLayerBuilder.nIn(hiddenLayerWidth)
        hiddenLayerBuilder.nOut(hiddenLayerWidth)
        // adopted activation function from GravesLSTMCharModellingExample
        // seems to work well with RNNs
        hiddenLayerBuilder.activation("tanh")
        listBuilder.layer(i, hiddenLayerBuilder.build())
    }

    // we need to use RnnOutputLayer for our RNN
    val outputLayerBuilder = RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
    // softmax normalizes the output neurons, the sum of all outputs is 1
    // this is required for our sampleFromDistribution-function
    outputLayerBuilder.activation("softmax");
    outputLayerBuilder.nIn(hiddenLayerWidth);
    outputLayerBuilder.nOut(numberOfLabels);
    listBuilder.layer(hiddenLayerCount + 1, outputLayerBuilder.build());

    // finish builder
    listBuilder.pretrain(false);
    listBuilder.backprop(true);
    //listBuilder.build();
    return listBuilder.build();
}

fun readDataset(inputPath: String, inputWidth: Int, dataName: String, numberOfExamples: Int, numberOfLabels: Int, miniBatchSize: Int): DataSetIterator {
    val featureReader: SequenceRecordReader = CSVSequenceRecordReader(0, ",")
    featureReader.initialize(NumberedAdhocFileInputSplit("$inputPath/nc${inputWidth}_${dataName}_feature_%d.csv", 1, 1301, numberOfExamples))
    val labelReader:SequenceRecordReader = CSVSequenceRecordReader(0, ",")
    labelReader.initialize(NumberedAdhocFileInputSplit("$inputPath/nc${inputWidth}_${dataName}_label_%d.csv", 1, 1301, numberOfExamples))

    return SequenceRecordReaderDataSetIterator(
            featureReader, labelReader,
            miniBatchSize,
            numberOfLabels, false,
            SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)
}
