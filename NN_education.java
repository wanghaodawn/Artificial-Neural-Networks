import java.io.*;
import java.util.*;
import java.util.Random;

public class NN_education {

    // Global variables
    private static int row;
    private static int col;
    private static String[][] sss;
    private static double[][] data;
    private static double[][] test;

    // Parameter of neural network
    private static final int num_hidden = 4;
    private static final int num_input = 6;
    private static final double step = 0.01;
    private static final double training_threshold = 1.3;

    // Weight of neural network
    private static double[][] w1 = new double[num_hidden][num_input]; 
    private static double[] w2 = new double[num_hidden];
    private static double error = Double.MAX_VALUE;
    private static double error_prev = 0.0;

	public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println("Incorrect Arguments");
            System.exit(-1);
        }
		
        // Read and preprocessing training data
        readData(args[0]);
 
        // Normalize data
        dataNormalization(true);
        
        // Set random weight
        setRandomWeight();        

        // Train Data
        do {
            error_prev = error;
            trainData();

        } while (error > training_threshold && error_prev >= error);

        System.out.println("TRAINING COMPLETED! NOW PREDICTING.");

        // Test data
        // Read and preprocessing training data
        readData(args[1]);
 
        // Normalize data
        dataNormalization(false);

        // Test data
        testData();
	}

    private static void testData() {
        for (int i = 0; i < row; i++) {
            // Compute hidden layer
            double[] data_hidden = new double[num_hidden];

            for (int j = 0; j < num_hidden; j++) {
                for (int k = 0; k < num_input; k++) {
                    data_hidden[j] += test[i][k] * w1[j][k];
                }
                // Sigmoid
                data_hidden[j] = sigmoid(data_hidden[j]);
            }

            // Compute output
            double output = 0.0;
            for (int j = 0; j < num_hidden; j++) {
                output += data_hidden[j] * w2[j];
            }
            output = sigmoid(output) * 100.0;
            System.out.println(output);
        }
    }

    private static void trainData() {

        error = 0.0;
        // Train the data in the whole training dataset
        for (int i = 0; i < row; i++) {
            
            // Compute hidden layer
            double[] data_hidden = new double[num_hidden];
            
            for (int j = 0; j < num_hidden; j++) {
                for (int k = 0; k < num_input; k++) {
                    data_hidden[j] += data[i][k] * w1[j][k];
                }
                // Sigmoid
                data_hidden[j] = sigmoid(data_hidden[j]);
            }

            // Compute output
            double output = 0.0;
            for (int j = 0; j < num_hidden; j++) {
                output += data_hidden[j] * w2[j];
            }
            output = sigmoid(output);

            // Update w2
            // System.out.println("data: " + data[i][num_input]);
            double delta = output * (1 - output) * (data[i][num_input] - output);
            // System.out.println("delta: " + delta);
            for (int j = 0; j < num_hidden; j++) {
                w2[j] += step * delta * data_hidden[j];
            }

            // Update w1
            for (int j = 0; j < num_hidden; j++) {
                double delta2 = data_hidden[j] * (1 - data_hidden[j]);
                for (int k = 0; k < num_input; k++) {
                    w1[j][k] += step * delta2 * w2[j] * delta * data[i][k];
                }
            }
            error += (output - data[i][num_input]) * (output - data[i][num_input]);
        }

        System.out.println(error);
    }

    private static void setRandomWeight() {
        // w1
        // Get random seed
        long seed = System.currentTimeMillis();
        // System.out.println("w1 seed: " + seed);
        Random random = new Random(seed);
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_input; j++) {
                w1[i][j] = random.nextDouble() % 1.0 - 0.5;
                // System.out.println(w1[i][j]);
            }
        }
        // w2
        // Get random seed
        seed = System.currentTimeMillis();
        // random = new Random(seed);
        // System.out.println("w2 seed: " + seed);
        for (int i = 0; i < num_hidden; i++) {
            w2[i] = random.nextDouble() % 1.0 - 0.5;
            // System.out.println(w2[i]);
        }
    }

    private static void dataNormalization(boolean isTrainingData) {

        double[][] aaa = new double[row][col+1];
        // double[] max_array = new double[col];
        // double[] min_array = new double[col];
        
        // min_array initialization
        // for (int i = 0; i < col; i++) {
        //     min_array[i] = Double.MAX_VALUE;
        // }

        // Convert string to double and find max and min for each column
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (sss[i][j].equals("yes")) {
                    aaa[i][j] = 1.0;
                } else if (sss[i][j].equals("no")) {
                    aaa[i][j] = 0.0;
                } else {
                    aaa[i][j] = Double.parseDouble(sss[i][j]);
                } 

                // max_array[j] = Math.max(max_array[j], aaa[i][j]);
                // min_array[j] = Math.min(min_array[j], aaa[i][j]);
            }
        }

        // double[] range_array = new double[col];
        // for (int i = 0; i < col; i++) {
        //     range_array[i] = max_array[i] - min_array[i];
        // }

        // Normalize the 2D matrix
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                aaa[i][j] = aaa[i][j] / 100.0;
            }
        }

        // System.out.print("\n");
        // // Test 2D matrix
        // for (int i = 0; i < row; i++) {
        //     for (int j = 0; j < col; j++) {
        //         System.out.print(aaa[i][j] + "\t");
        //     }
        //     System.out.print("\n");
        // }

        // System.out.print("max_array");
        // for (int i = 0; i < col; i++) {
        //     System.out.print(max_array[i] + "\t");
        // }
        // System.out.print("\n");

        // System.out.print("min_array");
        // for (int i = 0; i < col; i++) {
        //     System.out.print(min_array[i] + "\t");
        // }
        // System.out.print("\n");

        if (isTrainingData) {
            data = new double[row][col+1];
            // add a column of 1 at the begining of each column
            for (int i = 0; i < row; i++) {
                data[i][0] = 1.0;
                for (int j = 1; j < col+1; j++) {
                    data[i][j] = aaa[i][j-1];
                }
            }
        } else {
            test = new double[row][col+1];
            // add a column of 1 at the begining of each column
            for (int i = 0; i < row; i++) {
                test[i][0] = 1.0;
                for (int j = 1; j < col+1; j++) {
                    test[i][j] = aaa[i][j-1];
                }
            }
        }
        
        // System.out.println("data col: " + data[0].length);

        // System.out.print("\n");
        // // Test 2D matrix
        // for (int i = 0; i < row; i++) {
        //     for (int j = 0; j < col+1; j++) {
        //         System.out.print(data[i][j] + "\t");
        //     }
        //     System.out.print("\n");
        // }
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-1.0 * x));
    }

    private static void readData(String fileName) throws Exception {

        // Read training data
        File inFile = new File(fileName);
        
        // If file doesnt exists, then create it
        if (!inFile.exists()) {
            System.err.println("No file called: " + fileName);
            System.exit(-1);
        }

        BufferedReader br = null;
        StringBuilder sb = new StringBuilder();

        // Read string from the input file
        String sCurrentLine;
        
        br = new BufferedReader(new FileReader(inFile));

        // Throw the first line to ignore the name of attributes
        br.readLine();

        while ((sCurrentLine = br.readLine()) != null) {
            sb.append(sCurrentLine + "!");
            // Use . to mark the end of each line
        }

        String s = sb.toString();

        String[] ss = s.split("!");

        // Get the number of columns
        row = ss.length;
        // System.out.println("row: " + row);

        s = s.replaceAll("!", ",");
        ss = s.split(",");
        col = ss.length / row;

        // System.out.println("col: " + col);
        // System.out.println(s);
        // System.out.println("ss_length: " + ss.length);

        sss = new String[row][col];

        // Change array to 2D matrix
        for (int i = 0; i < ss.length; i++) {
            sss[i/col][i%col] = ss[i];
        }

        // Test 2D matrix
        // for (int i = 0; i < row; i++) {
        //     for (int j = 0; j < col; j++) {
        //         System.out.print(sss[i][j] + "\t");
        //     }
        //     System.out.print("\n");
        // }

    }
}