/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.chapter2.pytroch;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import static java.lang.System.out;
import java.util.Arrays;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Slice;
import org.bytedeco.pytorch.SymInt;
import org.bytedeco.pytorch.SymIntOptional;
import org.bytedeco.pytorch.TensorArrayRef;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexArrayRef;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import static org.bytedeco.pytorch.global.torch.cat;

/**
 *
 * @author omar
 */
public class DataManipulation {

  public static void main(String[] args) {

    Tensor X = torch.arange(new Scalar(0), new Scalar(12), new Scalar(1));
    torch.print(X);
    long dmnm = 0;
    out.println("Tensor Size:" + X.size(dmnm));
    out.println("Tensor Shape:" + Arrays.toString(X.shape()));

    X = X.resize_(3, 4);

    torch.print(X);
    out.println("----");
    Tensor zeros = torch.zeros(2, 3, 4);
    torch.print(zeros);
    out.println("----");
    Tensor ones = torch.ones(2, 3, 4);
    torch.print(ones);
    out.println("----");
    Tensor rand = torch.rand(3, 4);
    torch.print(rand);

    IntPointer intPointer = new IntPointer(2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1);
    Tensor tensor = torch.from_blob(intPointer, new long[]{3, 4}, new TensorOptions(torch.ScalarType.Int));
    torch.print(tensor);
    out.println("----");

    for (int i = 0; i < tensor.numel(); i++) {
      out.println("TensorArray " + i + ": " + tensor.data_ptr_int().get(i));
    }
    //Tensor sliced=x.slice(1, null, null, 3);
    Tensor sliced = X.slice(0, new LongOptional(1), new LongOptional(3), 1);
    out.print("----");
    torch.print(sliced);
    //Skipping index by -1 I have no idea how to do it
    Tensor slicedbyminus1 = X.select(0, -1);
    torch.print(slicedbyminus1);
    out.println("Index Opreations ----");

    //write elements of a matrix by specifying indices using index_put_ //  note this chnages the original tensor  
    //Copying the tensor using clone and maybe detach;
    Tensor tensor_index = new Tensor();
    tensor_index = X.clone().detach();
    torch.print(tensor_index.index_put_(new TensorIndexArrayRef(new TensorIndexVector(new TensorIndex(1), new TensorIndex(1))), new Scalar(17)));
    out.println(" multiple index opreation ----");
    //assign for multiple index using TensorIndexVector  //TensorIndex //Slice
    torch.print(tensor_index.index_put_(new TensorIndexArrayRef(new TensorIndexVector(new TensorIndex(new Slice(new SymIntOptional(new SymInt(0)), new SymIntOptional(new SymInt(2)), new SymIntOptional(new SymInt(1)))))), new Scalar(12)));

    //2.1.3. Operations
    out.println("Opreation----");
    torch.print(torch.exp(X));
    out.println(" arithmetic  Opreation----");
    IntPointer x_pointer = new IntPointer(1, 2, 4, 8);
    IntPointer y_pointer = new IntPointer(2, 2, 2, 2);

    Tensor x = torch.from_blob(x_pointer, new long[]{4}, new TensorOptions(torch.ScalarType.Int));
    Tensor y = torch.from_blob(y_pointer, new long[]{4}, new TensorOptions(torch.ScalarType.Int));
    torch.print(x);
    torch.print(y);
    out.println(" arithmetic  Opreation---- + - * / %");
    torch.print(torch.add(x, y));
    torch.print(torch.subtract(x, y));
    torch.print(torch.multiply(x, y));
    torch.print(torch.divide(x, y));
    torch.print(torch.pow(x, y));

    x = torch.arange(new Scalar(0), new Scalar(12), new TensorOptions(torch.ScalarType.Float)).reshape(3, 4);
    FloatPointer fp = new FloatPointer(2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1);
    y = torch.from_blob(fp, new long[]{3, 4}, new TensorOptions(torch.ScalarType.Float));
    torch.print(x);
    torch.print(y);
    //Concatenate multiple tensors
    out.println("Concatenate X,Y");
    torch.print(cat(new TensorArrayRef(new TensorVector(x, y)), 0));
    torch.print(cat(new TensorArrayRef(new TensorVector(x, y)), 1));
    //X==Y
    out.println("X==Y");
    torch.print(torch.eq(x, y));
    out.println("Sum X");
    torch.print(torch.sum(x));

    //2.1.4. Broadcasting
    out.println("2.1.4. Broadcasting------");
    out.println("a,b------");
    Tensor a = torch.arange(new Scalar(0), new Scalar(3), new TensorOptions(torch.ScalarType.Float)).reshape(3, 1);
    Tensor b = torch.arange(new Scalar(0), new Scalar(2), new TensorOptions(torch.ScalarType.Float)).reshape(1, 2);
    torch.print(a);
    torch.print(b);
    //a + b
    out.println("a + b------");
    torch.print(torch.add(a, b));
    out.println("---a a.item.float---");
    Tensor i = torch.tensor(3.5);
    torch.print(i);
    out.println(i.item().toFloat());
    out.println(" arithmetic  Opreation---- x==Y x<y x>y");
    torch.print(torch.eq(x, y));
    torch.print(torch.lessThan(x, y));
    torch.print(torch.greaterThan(x, y));
 

  }

}
