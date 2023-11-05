/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.chapter2.pytroch;

import static java.lang.System.out;

import java.util.function.Function;

import org.bytedeco.pytorch.BoolOptional;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.GeneratorOptional;
import org.bytedeco.pytorch.LongArrayRef;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorArrayRef;
import org.bytedeco.pytorch.TensorListOptional;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

/**
 *
 * @author omar
 */
public class AutomaticDifferentiation {

  public static Tensor f(Tensor a) {
    Tensor b = a.multiply(new Scalar(2));
    Tensor c;
    while (b.norm().item_float() < 1000) {
      b.multiplyPut(new Scalar(2));
      if (b.sum().item_float() > 0) {
        c = b;
      } else {
        c = b.multiply(new Scalar(100));
      }

    }
    b.close();
    return a;
  }

  public static void main(String[] args) {
    // https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout
    // torch.device(null);
    DeviceOptional device = new DeviceOptional(new Device(torch.DeviceType.CUDA));
    ScalarType scalarType = torch.ScalarType.BFloat16;
    TensorOptions tensorOptions = new TensorOptions();

    // initializing new TensorOptions objects with contructors wont work but using
    // as a factory works fine.
    TensorOptions tensorOption = tensorOptions.device(device).dtype(new ScalarTypeOptional(scalarType)).requires_grad(new BoolOptional(true));
    tensorOptions = null;
    out.println("tensorOption is_cuda:" + tensorOption.device().is_cuda());
    out.println("tensorOption is BFloat16 ? :" + tensorOption.dtype().isScalarType(scalarType));
    out.println("requires_grad is  ? :" + tensorOption.requires_grad());

    Tensor x = torch.arange(new Scalar(4.0), tensorOption).requires_grad_(true);
 
    // x=x.to(new Device(torch.DeviceType.CUDA), torch.ScalarType.BFloat16);

    out.println("x grad" + x.requires_grad());

    torch.print(x);

    Tensor y = torch.multiply(torch.dot(x, x), new Scalar(2)).requires_grad_(true);

    torch.print(y);
    y.backward();
    out.println("-----");
    torch.print(x.grad());

    // x.grad == 4 * x
    torch.print(torch.eq(x.grad(), x.multiply(new Scalar(4))));

    out.println("Restting grad");

    // reset grad
    x.grad().zero_();

    y = x.sum();
    y.backward();
    torch.print(x.grad());

    out.println("2.5.2. Backward for Non-Scalar Variables");

    // reset grad
    out.println("Is Y CUDA? " + y.device().is_cuda());
    x.grad().zero_();

    y = torch.multiply(x, x);

    Tensor z = torch.ones(new long[] { y.sizes().get(0) }, tensorOption);

    // torch.print(zz);
    y.backward(z, null, true, null);
    torch.print(x.grad());
    x.grad().zero_();

    out.println("2.5.3. Detaching Computation");
    out.println("-----");
    y = torch.multiply(x, x);
    Tensor u = y.detach();
    out.println("u:");
    torch.print(u);
    z = torch.multiply(u, x);
    out.println("z:");
    torch.print(z);
    z.sum().backward();
    out.println("x.gard:");
    torch.print(x.grad());
    out.println("x.grad == u");
    torch.print(x.grad().eq(u));
    out.println("------");
    x.grad().zero_();
    y.sum().backward();
    out.println("x.grad");
    torch.print(x.grad());
    out.println("x*2");
    torch.print(x.multiply(new Scalar(2)));

    out.println("2.5.4. Gradients and Python.. Control Flow");
    Tensor a = torch.randn(new long[]{},new GeneratorOptional(),tensorOption).requires_grad_(true);
    Tensor d=f(a);
    d.backward();

    torch.print(torch.eq(a.grad(), d.div(a)));
    
    System.exit(0);
  }

}
