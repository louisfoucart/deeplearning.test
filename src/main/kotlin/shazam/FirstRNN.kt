package shazam

import org.canova.api.records.reader.SequenceRecordReader
import org.canova.api.records.reader.impl.CSVSequenceRecordReader
import org.canova.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator

fun main(args: Array<String>) {
    val miniBatchSize = 100
    val numberOfLabels = 10
    val numberOfExamples = 1301
    val inputPath = "/src/main/resources/shazam/data_exploded"

    val featureReader: SequenceRecordReader = CSVSequenceRecordReader(0, ",")
    featureReader.initialize(NumberedFileInputSplit("$inputPath/nc13_train_feature_%d.csv", 1, numberOfExamples))
    val labelReader:SequenceRecordReader = CSVSequenceRecordReader(0, ",")
    labelReader.initialize(NumberedFileInputSplit("$inputPath/nc13_train_label_%d.csv", 1, numberOfExamples))

    val datasetIterator: DataSetIterator = SequenceRecordReaderDataSetIterator(
            featureReader, labelReader,
            miniBatchSize,
            numberOfLabels, false,
            SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

}
