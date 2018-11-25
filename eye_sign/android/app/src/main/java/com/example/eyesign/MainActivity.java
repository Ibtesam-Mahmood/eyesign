package com.example.eyesign;

import android.os.Bundle;
import io.flutter.app.FlutterActivity;
import io.flutter.plugins.GeneratedPluginRegistrant;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends FlutterActivity {
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    GeneratedPluginRegistrant.registerWith(this);
    doDataRetrieval();
  }

  String modelFile="asl.lite";
  Interpreter tflite;

  private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {

    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();

    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();

    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public void doDataRetrieval()
  {
    try {
      tflite=new Interpreter(loadModelFile(MainActivity.this,modelFile));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public float[] runTensorflow(String imgpath)
  {
    float[] out=new float[]{0};
    tflite.run(imgpath,out);

    return out;
  }

  public String getASL(String imgpath)
  {
    float[] letter = runTensorflow(imgpath);
    int biggest = 0;

    for (int i = 0; i<letter.length;i++)
    {

      if (letter[i] > letter[biggest])
      {
        biggest = i;
      }
    }

    switch (biggest){
      case 0:
        return "a";
      break;
      case 1:
        return "b";
      break;
      case 2:
        return "c";
      break;
      case 3:
        return "d";
      break;
      default:
        return "this letter isn't added yet!";
      break;
    }
  }


}
