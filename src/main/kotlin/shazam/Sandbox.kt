package shazam


fun main(args: Array<String>) {
    //var maxWidth
    val reader = openReader("/shazam/data/nc13-train.csv")
    reader.readLine() // skip header
    reader.forEachLine {
        val columns = it.split(",")
        //columns[14]
    }
}