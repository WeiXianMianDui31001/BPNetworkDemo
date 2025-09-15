
public class Main {
    public static void main(String[] args) {
        BPNetworkDemo bpNetworkDemo = new BPNetworkDemo();
        double[] x = {0.5, -0.3, 0.8};
        double[] y = {1, 0, -1};
        for(int epoch = 0;epoch < 1000; epoch++) {
            bpNetworkDemo.forward(x);
            double L = bpNetworkDemo.loss(y);
            bpNetworkDemo.backward(y);
            if( epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " Loss:" + L);
            }
        }
        double[] out = bpNetworkDemo.forward(x);
        System.out.println("Final output:");
        for(double v: out) {
            System.out.println(v);
        }
    }
}
