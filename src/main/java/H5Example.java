import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class H5Example {
    public static void main(String[] args) throws  Exception {

        final String modelWeights = "saved_model.h5";
        final MultiLayerNetwork model;


        String modelWeightsPath = new ClassPathResource(modelWeights).getFile().getPath();
        System.out.println(modelWeightsPath);
        model = KerasModelImport.importKerasSequentialModelAndWeights(modelWeightsPath);


        //Create input INDArray for the user measurements
        INDArray actualInput = Nd4j.zeros(1,13);
        actualInput.putScalar(new int[]{0,0}, 63);
        actualInput.putScalar(new int[]{0,1}, 1);
        actualInput.putScalar(new int[]{0,2}, 3);
        actualInput.putScalar(new int[]{0,3}, 145);
        actualInput.putScalar(new int[]{0,4}, 233);
        actualInput.putScalar(new int[]{0,5}, 1);
        actualInput.putScalar(new int[]{0,6}, 0);
        actualInput.putScalar(new int[]{0,7}, 150);
        actualInput.putScalar(new int[]{0,8}, 0);
        actualInput.putScalar(new int[]{0,9}, 2.3);
        actualInput.putScalar(new int[]{0,10}, 0);
        actualInput.putScalar(new int[]{0,11}, 0);
        actualInput.putScalar(new int[]{0,12}, 1);
        INDArray prediction = model.output(actualInput);
        System.out.println("Prediction:"+prediction.toString());
    }

}
