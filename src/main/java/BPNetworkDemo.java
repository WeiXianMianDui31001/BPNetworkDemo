import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BPNetworkDemo {
    double relu(double x) {
        return x > 0 ? x : 0;
    }
    double relu_derivative(double x) {
        return x > 0 ? 1 : 0;
    }
    final static double lr = 0.01;
    ArrayList<Integer> layers = new ArrayList<>(Arrays.asList(3, 4, 4, 4, 3));
    ArrayList<double[][]> W = new ArrayList<>();
    ArrayList<double[]> b = new ArrayList<>();

    ArrayList<double[]> a = new ArrayList<>();
    ArrayList<double[]> z = new ArrayList<>();

    ArrayList<double[]> delta = new ArrayList<>();

    BPNetworkDemo() {
        Random rand = new Random(0);
        int L = layers.size();
        W.add(new double[0][0]);
        b.add(new double[0]);
        a.add(new double[0]);
        z.add(new double[0]);


        delta.add(new double[0]);

        for (int l = 1; l < L; l++) {
            int nodes = layers.get(l);
            int prev_nodes = layers.get(l - 1);
            W.add(new double[nodes][prev_nodes]);
            b.add(new double[nodes]);
            a.add(new double[nodes]);
            z.add(new double[nodes]);
            delta.add(new double[nodes]);
            for(int i = 0; i < nodes; i++) {
                for(int j = 0; j < prev_nodes; j++) {
                    W.get(l)[i][j] = rand.nextDouble() * 2 - 1;
                }
                b.get(l)[i] = 0;
            }
        }
    }


    double[] forward(double[] input) {
        a.set(0, input);
        int L = layers.size();
        for(int l = 1; l < L; l++) {
            for(int i = 0; i < layers.get(l); i++) {
                double sum = b.get(l)[i];
                for(int j = 0; j < layers.get(l - 1); j++) {
                    sum += W.get(l)[i][j] * a.get(l - 1)[j];
                }
                z.get(l)[i] = sum;
                if(l == L - 1) {
                    a.get(l)[i] = sum;
                }
                else {
                    a.get(l)[i] = relu(sum);
                }
            }
        }
        return a.get(L - 1);
    }

    double loss(double[] target) {
        double L = 0.0;
        int out_layer =layers.size() - 1;
        for(int i = 0; i < layers.get(out_layer); i++) {
            double diff =a.get(out_layer)[i] - target[i];
            L += diff * diff;
        }
        return L;
    }

    void backward(double[] target ) {
        int out_layer =layers.size() - 1;
        //输出层delta
        for(int i = 0; i < layers.get(out_layer); i++) {
            delta.get(out_layer)[i] = 2 * (a.get(out_layer)[i] - target[i]);
        }
        //隐藏层delta
        for (int l = layers.size() - 2; l >= 1; l-- ) {
            for(int i = 0; i < layers.get(l); i++) {
                double grad = 0;
                for(int k = 0; k < layers.get(l + 1); k++) {
                    grad += W.get(l + 1)[k][i] * delta.get(l + 1)[k];
                }
                delta.get(l)[i] = grad * relu_derivative(z.get(l)[i]);
            }
        }
        //更新权重和偏置
        for(int l = 1; l < layers.size(); l++) {
            for(int i = 0; i < layers.get(l); i++) {
                for(int j = 0; j < layers.get(l - 1); j++) {
                    W.get(l)[i][j] -= lr * delta.get(l)[i] * a.get(l - 1)[j];
                }
                b.get(l)[i] -= lr * delta.get(l)[i];
            }
        }
    }
}
