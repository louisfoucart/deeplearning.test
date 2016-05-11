package shazam

import java.io.File
import java.util.*

fun main(args: Array<String>) {
    File("data/nc13-train.csv").readLines().forEach {
        val columns = it.split(",")
        val output = getOutput(columns[0])
        (1..4).forEach { output.appendText(columns[it]) }
    }
}

val mapOfOutputs: Map<String, File> = HashMap()
fun getOutputFromMap(name: String): File {
    var output = mapOfOutputs[name]
    if(output == null) {
        output = File(name)
    }
    return output
}

var currentOutput: File = File("test")
var currentName: String = "test"
fun getOutput(name: String): File {
    if(name == currentName)
        return currentOutput
    currentOutput = File(name)
    currentName = name
    return currentOutput
}
