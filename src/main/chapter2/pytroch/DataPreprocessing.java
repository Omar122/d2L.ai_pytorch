package main.chapter2.pytroch;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import static java.lang.System.out;

/**
 *
 * @author omar
 */
public class DataPreprocessing {

  public static void main(String[] args) throws IOException {

    File file = new File("../data/");
    file.mkdir();

    String dataFile = "../data/house_tiny.csv";

    // Create file
    File f = new File(dataFile);
    f.createNewFile();

    // Write to file
    try (FileWriter writer = new FileWriter(dataFile)) {
      writer.write("NumRooms,Alley,Price\n"); // Column names
      writer.write("NA,Pave,127500\n"); // Each row represents a data example
      writer.write("2,NA,106000\n");
      writer.write("4,NA,178100\n");
      writer.write("NA,NA,140000\n");
    } catch (Exception e) {
      System.out.println("Error file wirter" + e.getMessage());
    }
    System.out.println("f:" + f.getCanonicalPath());
    Table data = Table.read().csv(f.getCanonicalPath());

    System.out.println(data.structure().printAll());

    System.out.println(data.printAll());

    Table input = data.create(data.columns());
    input.removeColumns("Price");

    Table output = data.selectColumns("Price");

    Column col = input.column("NumRooms");
    col.set(col.isMissing(), (int) input.nCol("NumRooms").mean());

    System.out.println("Output");
    System.out.println(output.printAll());
    System.out.println("Input");
    System.out.println(input.printAll());

    StringColumn colAlley = (StringColumn) input.column("Alley");
    List<BooleanColumn> dummies = colAlley.getDummies();
    input.removeColumns(colAlley);
    input.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
        DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray()));

    System.out.println("Input After creating dummies from Alley Column");
    System.out.println(input.printAll());

    double[][] InputMatrix = input.as().doubleMatrix();
    double[][] outputMatrix = output.as().doubleMatrix();
    // To collections or to arrays

    DoublePointer inputDoublePointer = new DoublePointer(
        Arrays.stream(InputMatrix).flatMapToDouble(Arrays::stream).toArray());
    DoublePointer outputDoublePointer = new DoublePointer(
        Arrays.stream(outputMatrix).flatMapToDouble(Arrays::stream).toArray());

    Tensor inputTensor = torch.from_blob(inputDoublePointer,
        new long[] { InputMatrix.length, InputMatrix[InputMatrix.length - 1].length },
        new TensorOptions(torch.ScalarType.Double));
    Tensor outputTensor = torch.from_blob(outputDoublePointer,
        new long[] { outputMatrix.length, outputMatrix[outputMatrix.length - 1].length },
        new TensorOptions(torch.ScalarType.Double));

    torch.print(inputTensor);
    out.println("=====");
    torch.print(outputTensor);
    out.println("=====");

    // Trying to add new columns based on number of rooms.
    IntColumn colRooms = (IntColumn) input.column("NumRooms");

    input.addColumns(DoubleColumn.create("Stdiuo", colRooms.asList().stream().map(e -> e <= 2 ? 1 : 0).toList()));
    input.addColumns(DoubleColumn.create("Wing", colRooms.asList().stream().map(e -> e > 2 ? 1 : 0).toList()));
    out.println("=====");
    out.println(input.printAll());
  }
}
