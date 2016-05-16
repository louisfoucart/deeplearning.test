package shazam

import org.slf4j.LoggerFactory
import java.io.BufferedReader
import java.io.File
import java.io.PrintWriter
import java.util.*

val log = LoggerFactory.getLogger(Anchor::class.java)

fun main(args: Array<String>) {
    generateTrainData("src/main/resources/shazam/data_train", "nc13")
    //generateValidationData("src/main/resources/shazam/data_validation", "nc13")
}

fun generateTrainData(destinationPath: String, dataName: String) {
    val type= "train"
    File(destinationPath).mkdirs()
    val reader = openReader("data/$dataName-${type}.csv")
    reader.readLine() // skip header
    reader.forEachLine {
        val columns = it.split(",")
        val soundId = removeQuotes(columns[0])
        val className = removeQuotes(columns[15])
        val featureOutput = getFeatureOutputAndWriteLabel(dataName, type, soundId, className, destinationPath)
        (1..12).forEach { // write first 12 bands separated by commas
            featureOutput.print(columns[it])
            featureOutput.print(",") }
        featureOutput.println(columns[13]) // write last band
    }
    //log.info("labelSize: ${classesSet.size}")
    writeLabelCodeMapToFile(destinationPath, dataName)
}

fun generateValidationData(destinationPath: String, dataName: String) {
    File(destinationPath).mkdirs()
    val reader = openReader("/shazam/data/$dataName-validation.csv")
    reader.readLine() // skip header
    reader.forEachLine {
        val columns = it.split(",")
        val soundId = removeQuotes(columns[0])
        val featureOutput = getFeatureOutput(dataName, "validation", soundId, destinationPath)
        (1..12).forEach { // write first 12 bands separated by commas
            featureOutput.print(columns[it])
            featureOutput.print(",") }
        featureOutput.println(columns[13]) // write last band
    }
}

fun removeQuotes(s: String): String {
    return s.substring(1, s.length - 1)
}

//these variables serve to implement iterative file generation, test file serves nothing, it is just to allow val init
var currentOutput = File("test").printWriter()
var currentName = "test"
fun getFeatureOutput(dataName: String, type: String, exampleName: String, destinationPath: String): PrintWriter {
    if(exampleName == currentName)
        return currentOutput
    currentOutput.close()
    val featureOutput = File("$destinationPath/${dataName}_${type}_feature_$exampleName.csv")
    log.info("feature file  ${featureOutput.absolutePath}")
    featureOutput.createNewFile()
    currentOutput = featureOutput.printWriter()
    currentName = exampleName
    return currentOutput
}

fun getFeatureOutputAndWriteLabel(dataName: String, type: String, exampleName: String, className: String, destinationPath: String): PrintWriter {
    if(exampleName == currentName)
        return currentOutput
    currentOutput.close()
    writeLabelOutput(dataName, type, exampleName, className, destinationPath)
    val featureOutput = File("$destinationPath/${dataName}_${type}_feature_$exampleName.csv")
    log.info("feature file  ${featureOutput.absolutePath}")
    featureOutput.createNewFile()
    currentOutput = featureOutput.printWriter()
    currentName = exampleName
    return currentOutput
}

//val classesSet: MutableSet<String> = HashSet()
fun writeLabelOutput(dataName: String, type: String, exampleName: String, className: String, destinationPath: String) {
    val labelOutput = File("$destinationPath/${dataName}_${type}_label_$exampleName.csv")
    log.info("label file  ${labelOutput.absolutePath}")
    labelOutput.createNewFile()
    labelOutput.writeText(getLabelCode(className))
    //classesSet.add(className)
}

val labelCodeMap:MutableMap<String, Int> = HashMap()
var nextLabelCode = 0
fun getLabelCode(className: String): String {
    if(labelCodeMap.containsKey(className)) {
        return labelCodeMap[className].toString()
    } else {
        val labelCode = nextLabelCode
        nextLabelCode++
        labelCodeMap.put(className, labelCode)
        return labelCode.toString()
    }
}

fun writeLabelCodeMapToFile(destinationPath: String, dataName: String) {
    val codeMap = File("$destinationPath/${dataName}_labelCodeMap.csv")
    codeMap.createNewFile()
    labelCodeMap.entries.forEach { codeMap.appendText("${it.key}=${it.value}\n") }
}

class Anchor

fun openReader(name: String): BufferedReader {
    return File(name).bufferedReader()
}