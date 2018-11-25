import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';


class CameraExampleHome extends StatefulWidget {
  @override
  _CameraExampleHomeState createState() => new _CameraExampleHomeState();
   
}




/// Returns a suitable camera icon for [direction]. SELECTOR FOR CAMERA
IconData getCameraLensIcon(CameraLensDirection direction) {
  switch (direction) {
    case CameraLensDirection.back:
      return Icons.camera_rear;
    case CameraLensDirection.front:
      return Icons.camera_front;
    case CameraLensDirection.external:
      return Icons.camera;
  }
  throw ArgumentError('Unknown lens direction');
}



void logError(String code, String message) =>
    print('Error: $code\nError Message: $message');

class _CameraExampleHomeState extends State<CameraExampleHome> 
{
  CameraController controller;

  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      
      body: Column(
        children: <Widget>[
          Expanded(
            child: Container(
              child: Padding(
                padding: const EdgeInsets.all(0.0),
                child: GestureDetector(
                  child:_cameraPreviewWidget(),
                  onDoubleTap: () {
                    flipCamera();
                  } ,
                )
              ),
            ),
          ),
          _captureControlRowWidget(),
          Padding(
            padding: const EdgeInsets.all(76.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: <Widget>[
              
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    controller = new CameraController(cameras[0], ResolutionPreset.high);
    controller.initialize().then((_)
    {
      if (!mounted)
      {
        return;
      }
    setState(() {
          
        });
    });
  }

  /// Display the preview from the camera (or a message if the preview is not available).
  Widget _cameraPreviewWidget() {
    if (controller == null || !controller.value.isInitialized) 
    { 
      
    } 
    else 
    {
      final size = MediaQuery.of(context).size;
      final deviceRatio = (size.width / (size.height -76));
      return Transform.scale(
        scale: controller.value.aspectRatio / deviceRatio,
        child: AspectRatio(
          aspectRatio: controller.value.aspectRatio,
          child: CameraPreview(controller),
        ),
      );
    }
  }
  /// Display the control bar with buttons to take pictures and record videos.
  Widget _captureControlRowWidget() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      mainAxisSize: MainAxisSize.max,
      children: <Widget>[
        
      
      ],
    );
  }

  /// Display a row of toggle to select the camera (or a message if no camera is available).
  /*Widget _cameraTogglesRowWidget() {
    final List<Widget> toggles = <Widget>[];
  {
   
      for (CameraDescription cameraDescription in cameras) {
        toggles.add(
          SizedBox(
            width: 164.0,
            child: RadioListTile<CameraDescription>(
              title: Icon(getCameraLensIcon(cameraDescription.lensDirection)),
              groupValue: controller?.description,
              value: cameraDescription,
              onChanged: controller != null && controller.value.isRecordingVideo
                  ? null
                  : onNewCameraSelected,
            ),
          ),
        );
      }
    }

    return Row(children: toggles);
  }*/
  bool _isRecording() {
    return controller != null;// && controller.value.isRecordingVideo
  }
  int _count = 0;
  Widget _doubleTapWidget() {
    CameraDescription cameraDescription;


    return GestureDetector(
      child: _cameraPreviewWidget(),
      onDoubleTap: () {
        _count++;
        _count = _count <= 1 ? _count : 0;

        cameraDescription = cameras.elementAt(_count);
        onNewCameraSelected(cameraDescription);
      },
    );
  }
  
Widget switchCamera(BuildContext context){
  return new GestureDetector(
    child: ButtonBar(),
    onDoubleTap: () => flipCamera(),
  );
}

Future<Null> _restartCamera(CameraDescription description) async {
    final CameraController tempController = controller;
    controller = null;
    await tempController?.dispose();
    controller = new CameraController(description, ResolutionPreset.high);
await controller.initialize();
}

Future<Null> flipCamera() async {
  if (controller != null) {
    var newDescription = cameras.firstWhere((desc) {
      return desc.lensDirection != controller.description.lensDirection;
    });

    await _restartCamera(newDescription);
  }
}





  String timestamp() => DateTime.now().millisecondsSinceEpoch.toString();

  void showInSnackBar(String message) {
    _scaffoldKey.currentState.showSnackBar(SnackBar(content: Text(message)));
  }

  void onNewCameraSelected(CameraDescription cameraDescription) async {
    if (controller != null) {
      await controller.dispose();
    }
    controller = CameraController(cameraDescription, ResolutionPreset.high);

    // If the controller is updated then update the UI.
    controller.addListener(() {
      if (mounted) setState(() {});
      if (controller.value.hasError) {
        showInSnackBar('Camera error ${controller.value.errorDescription}');
      }
    });

    try {
      await controller.initialize();
    } on CameraException catch (e) {
      
    }

    if (mounted) {
      setState(() {});
    }
  }
}

class CameraApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CameraExampleHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

List<CameraDescription> cameras;

Future<Null> main() async {
  cameras = await availableCameras();
  runApp(new CameraApp());
}
