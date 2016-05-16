package util;

import org.canova.api.split.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by louis on 14/05/2016.
 */
public class NumberedAdhocFileInputSplit implements InputSplit {
    private final String baseString;
    private final int minIdx;
    private final int maxIdx;
    private final int numberOfExamples;

    /**
     * @param baseString String that defines file format. Must contain "%d", which will be replaced with
     *                   the index of the file.
     * @param minIdxInclusive Minimum index/number (starting number in sequence of files, inclusive)
     * @param maxIdxInclusive Maximum index/number (last number in sequence of files, inclusive)
     */
    public NumberedAdhocFileInputSplit(String baseString, int minIdxInclusive, int maxIdxInclusive, int numberOfExamples){
        if(baseString == null || !baseString.contains("%d")){
            throw new IllegalArgumentException("Base String must contain  character sequence %d");
        }
        this.baseString = baseString;
        this.minIdx = minIdxInclusive;
        this.maxIdx = maxIdxInclusive;
        this.numberOfExamples = numberOfExamples;
    }

    @Override
    public long length() {
        throw new UnsupportedOperationException();
    }

    @Override
    public URI[] locations() {
        int length = 0;
        List<URI> uris = new ArrayList(maxIdx-minIdx+1);
        for( int i=minIdx; i<=maxIdx && length < numberOfExamples; i++ ){
            Path path = Paths.get(String.format(baseString, i));
            if(path.toFile().exists()) {
                uris.add(path.toUri());
                length++;
            }
        }
        return uris.toArray(new URI[length]);
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    @Override
    public double toDouble(){
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat(){
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt(){
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong(){
        throw new UnsupportedOperationException();
    }

}
