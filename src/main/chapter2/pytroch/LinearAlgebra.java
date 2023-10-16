/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.chapter2.pytroch;

import java.util.Arrays;
import static java.lang.System.*;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.pytorch.BFloat16;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

/**
 *
 * @author omar
 */
public class LinearAlgebra {

  public static void main(String[] args) {

    Tensor x = torch.tensor(3);
    Tensor y = torch.tensor(2);

    torch.print(x);
    torch.print(y);

    out.println("X+Y: ");
    torch.print(x.add(y));

    out.println("X*Y: ");
    torch.print(x.multiply(y));
    out.println("X/Y: ");
    torch.print(x.div(y));
    out.println("X**Y: ");
    torch.print(x.pow(y));

    //Vectors 
    out.println("====");
    out.println("Vectors");
    Tensor vector = torch.arange(new Scalar(0), new Scalar(3), new Scalar(1));

    torch.print(vector);

    out.print("Vector 2nd Element: ");
    //vector.slice(dims,start,end,step)
    out.println(vector.data_ptr_long().get(2));
    out.println("---");
    out.println("Vextor Length: " + vector.size(0));
    out.print("Vector Shape: " + Arrays.toString(vector.shape()));

    //Matrices
    out.println("====");
    out.println("Matrices");
    Tensor matrixThreeByTwo = torch.arange(new Scalar(0), new Scalar(6), new Scalar(1)).reshape(3, 2);
    torch.print(matrixThreeByTwo);
    out.println("matrix 2BY3 Transpose ");
    torch.print(matrixThreeByTwo.t());

    //Matrix are  equal to their T // A==A.T if they symmetric 
    out.println("A==A.T 3,3 Matrix");
    Tensor symmetricMatrix = torch.from_blob(new IntPointer(1, 2, 3, 2, 0, 4, 3, 4, 5), new long[]{3, 3}, new TensorOptions(torch.ScalarType.Int));
    torch.print(symmetricMatrix.t().eq(symmetricMatrix));

    out.println("====");

    //Tnsors 
    out.println("Tnsors");
    Tensor tensor = torch.arange(new Scalar(0), new Scalar(24), new Scalar(1)).reshape(2, 3, 4);
    torch.print(tensor);

    out.println("---");
    //Basic Properties of Tensor Arithmetic
    Tensor a = torch.arange(new Scalar(0), new Scalar(6), new Scalar(1), new TensorOptions(torch.ScalarType.BFloat16)).reshape(2, 3);
    out.println("---\n");
    out.println("Matrix a:");
    torch.print(a);

    Tensor b = null;
    b = a.clone();

    out.println("Matrix b:");
    torch.print(b);
    out.println("a+b: ");
    torch.print(a.add(b));
    out.println("a*b: ");
    torch.print(a.multiply(b));
    out.println("--");
    //Tensor * scalar ;
    out.println("2+Tensor: ");
    torch.print(tensor.add(new Scalar(2)));
    out.print("2*Tensor Szie: ");
    out.println(Arrays.toString(tensor.multiply(new Scalar(2)).shape()));
    out.println("---");
    out.println("---");

    //Reduction
    out.println("Reduction: ");
    out.println("Vectors: a");
    torch.print(vector);
    out.println("vertor a Sum: ");
    torch.print(torch.sum(vector));
    out.println("===");

    out.println("Matrix Shape:" + Arrays.toString(a.shape()));
    out.println("Matrix Sum: ");
    torch.print(torch.sum(a));
    out.println("Matrix Sum axis 0: ");
    torch.print(torch.sum(a, 0));
    out.println("===");
    out.println("Matrix Sum axis 1: ");
    torch.print(torch.sum(a, 1));
    out.println("===");
    out.println("Sum of a axis 0,1 == a.sum: ");
    torch.print(torch.sum(a, 0, 1).eq(torch.sum(a)));

    out.println("\n===");

    //Matrix Mean == Matrix Sum/Size
    out.println("Matrix Mean" + torch.mean(a).item_float());
    out.println("Matrixc Sum/size" + torch.sum(a).item_float() / a.numel());

    //Reduceing a tensor along specific axes
    out.println("Matrix Mean along axis 0");
    torch.print(torch.mean(a, 0));
    out.println("Matrix Sum/size == mean along axis 0");
    torch.print(torch.sum(a, 0).div(new Scalar(a.size(0))));
    out.println("\n===");

    //Non-Reduction Sum 
    out.println("Non-Reduction Sum");

    out.println("Sum a with keepdims True allong axis 0: ");
    Tensor sum_a = torch.sum(a, new long[]{1}, true, new ScalarTypeOptional(torch.ScalarType.BFloat16));
    torch.print(sum_a);
    out.println("\n A/a_sum");
    torch.print(a.div(sum_a));
    out.println("\n===");
    //out.println("Sum a / Sum Sof A axis 1: //Also known as boardcast" + NDArray.broadcast_div(a, NDArray.sum_axis(p1)[0], null)[0].toString());
    out.println("Cumulative sum: ");
    torch.print(torch.cumsum(a, 0));
    out.print("\n===");
    out.print("===");
    //Dot Products
    out.println("X: ");
    x = torch.arange(new Scalar(0), new Scalar(3), new Scalar(1), new TensorOptions(torch.ScalarType.BFloat16));
    y = torch.ones(new long[]{3}, new TensorOptions(torch.ScalarType.BFloat16));
    torch.print(x);
    out.println("Y: ");
    torch.print(y);

    out.println("===");
    out.println("y . x : " + x.dot(y).item().toFloat());
    out.println("Sum of x * y: " + torch.sum(x.multiply(y)).item().toFloat());
    out.println("===");

    //Matrix–Vector Products
    out.println("Matrix–Vector Products");
    //Eexpress a matrix–vector product in code, we use the mv function. Note that the column dimension of A (its length along axis 1) must be the same as the dimension of x (
    out.println("A Shapre" + Arrays.toString(a.shape()) + " X Shapre:" + Arrays.toString(x.shape()));
    out.println("A move x : ");
    torch.print(torch.mv(a, x));

    //Matrix–Matrix Multiplication
    out.println("Matrix–Vector Products");
    Tensor B = torch.ones(new long[]{3, 4}, new TensorOptions(torch.ScalarType.BFloat16));
    torch.print(torch.mm(a, B));
    out.println("===");

    //Norms
    out.println("\nNorms");
    //https://github.com/bytedeco/javacpp-presets/issues/1425 
    ShortPointer f16Array = new ShortPointer(new BFloat16(3F).x(), new BFloat16(-4F).x());
    /*
    float[] fa={3,-4};
    short[] bfa=new short[fa.length];
    for (int i = 0; i < fa.length; i++) {
      out.print("1");
      bfa[i]=new BFloat16(fa[i]).x(); 
    }
   //OR
    Tensor t1 = torch.from_blob(new FloatPointer(3, -4), new long[]{2}, new TensorOptions(torch.ScalarType.Float));
    Tensor u = t1.to(ScalarType.BFloat16);
     */

    Tensor u = torch.from_blob(f16Array, new long[]{2}, new TensorOptions(torch.ScalarType.BFloat16));
    torch.print(u);
    out.println("===");
    out.print("Norm 1 of 3,-4 Tensor: ");
    out.println(torch.norm(u).item_float());
    out.println("===");
    out.print("Norm 2 of 3,-4 Tensor: ");
    out.println(torch.abs(u).sum().item_float());

    out.println("The Frobenius norm of one(4,9)  " + torch.norm(torch.ones(4, 9)).item_float());
    
    out.println("===");
    out.println("Exercises");
    //Exercises
    out.println("At+Bt=(A+B)t");
    //At+Bt=(A+B)t
    Tensor atPlusBt = a.t().add(b.t());
    Tensor aplusB = a.add(b).t();
    torch.print(atPlusBt.eq(aplusB));

    exit(0);

  }

}
