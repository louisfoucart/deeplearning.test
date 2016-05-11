package shazam

import org.slf4j.LoggerFactory
import java.io.BufferedReader
import java.io.File
import java.io.PrintWriter
import java.util.*

val log = LoggerFactory.getLogger(Anchor::class.java)

fun main(args: Array<String>) {
    val destinationPath = "src/main/resources/shazam/data_exploded"
    File(destinationPath).mkdirs()

    val reader = openReader("/shazam/data/nc13-train.csv")
    reader.readLine() // skip header
    reader.forEachLine {
        //log.debug(it)
        val columns = it.split(",")
        val featureOutput = getFeatureOutputAndWriteLabelOutput(removeQuotes(columns[0]), columns[15], destinationPath)
        (1..12).forEach { // write first 12 bands separated by commas
            featureOutput.print(columns[it])
            featureOutput.print(",") }
        featureOutput.println(columns[13]) // write last band
    }
    log.info("labelSize: ${classesSet.size}")
}

fun removeQuotes(s: String): String {
    return s.substring(1, s.length - 1)
}

val mapOfOutputs: Map<String, File> = HashMap()
fun getFeatureOutputAndWriteLabelOutputWithMap(name: String, className: String): File {
    var output = mapOfOutputs[name]
    if(output == null) {
        output = File(name)
    }
    return output
}

var currentOutput = File("test").printWriter()
var currentName = "test"
val classesSet: MutableSet<String> = HashSet()
fun getFeatureOutputAndWriteLabelOutput(name: String, className: String, destinationPath: String): PrintWriter {
    if(name == currentName)
        return currentOutput
    currentOutput.close()
    val featureOutput = File("$destinationPath/nc13_train_feature_$name.csv")
    log.info("feature file  ${featureOutput.absolutePath}")
    featureOutput.createNewFile()
    currentOutput = featureOutput.printWriter()
    currentName = name
    val labelOutput = File("$destinationPath/nc13_train_label_$name.csv")
    log.info("label file  ${labelOutput.absolutePath}")
    labelOutput.createNewFile()
    labelOutput.writeText(className)
    classesSet.add(className)
    return currentOutput
}

class Anchor

fun openReader(name: String): BufferedReader {
    return Anchor::class.java.getResourceAsStream(name).bufferedReader()
}