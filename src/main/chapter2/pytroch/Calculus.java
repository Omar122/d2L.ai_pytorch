/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.chapter2.pytroch;

import java.io.IOException;
import static java.lang.System.out;
import java.net.URISyntaxException;
import java.util.function.Function;
import main.util.PlotlyUtil;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

/**
 *
 * @author omar
 */
public class Calculus {

  public static void main(String[] args) throws IOException, URISyntaxException {
    Function<Double, Double> f = x -> (3 * Math.pow(x, 2) - 4 * x);
    Function<Double, Double> numLimit = x -> ((f.apply(x + 1) - f.apply(1.0)) / x);

    PlotlyUtil plotlyUtil = new PlotlyUtil();



    //NDArray x = NDArray.arange(-1, -6, -1, 1, context, DType.Float64());
    Tensor x= torch.arange(new Scalar(-1), new Scalar(-6),new Scalar(-1)).to(torch.ScalarType.BFloat16);
    
    out.println("X Tensor");
    torch.print(x);
    
    
    double[] d = new double[(int)x.numel()];

    for (int i = 0; i < d.length; i++) {
      d[i] = Math.pow(10, x.get(i).item_short());
    }

    for (int i = 0; i < 5; i++) {
      out.printf("h: %.5f", d[i]);
      out.printf(" numerical limit: %.5f %n", numLimit.apply(d[i]));
    }
  
    //Visualization Utilities
    x= torch.arange(new Scalar(0), new Scalar(3),new Scalar(0.1)).to(torch.ScalarType.BFloat16);
    

    double[] fx = new double[(int)x.numel()];
    
    for (int i = 0; i < x.numel(); i++) {
      fx[i] = f.apply((double)x.get(i).item_double());
    }

    double[] fg = new double[(int)x.numel()];
    for (int i = 0; i < x.numel(); i++) {
      fg[i] = 2 * x.get(i).item_double()- 3;
    }
    
    double[] xArray = new double[(int)x.numel()];
    for (int i = 0; i < (int)x.numel(); i++) {
      xArray[i]=x.get(i).item_double();
    }

    Figure figure = plotlyUtil.plotLineAndSegment(
        xArray, fx, fg, "f(x)", "Tangent line(x=1)", "x", "f(x)", 1000, 800);
    //On ubuntu without genome dont forget to install "desktop-file-utils" and run the command. 
    //sudo apt install desktop-file-utils
    //update-desktop-database
    Plot.show(figure);
    
  }

}
