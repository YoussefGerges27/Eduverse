import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import '../services/exam_service.dart';
import 'view_examas_screen.dart';
import 'package:google_mlkit_face_mesh_detection/google_mlkit_face_mesh_detection.dart';
import 'package:permission_handler/permission_handler.dart';

late List<CameraDescription> cameras;

class EnglishExamScreen extends StatefulWidget {
  final int examId;
  const EnglishExamScreen({super.key, this.examId = 1});

  @override
  State<EnglishExamScreen> createState() => _EnglishExamScreenState();
}

class _EnglishExamScreenState extends State<EnglishExamScreen> {
  CameraController? _cameraController;
  bool _cameraReady = false;
  bool _isLoading = true;
  String? _errorMessage;

  late Map<String, dynamic> _examData;
  late List<dynamic> _questions;
  int _currentQuestionIndex = 0;
  Map<int, int> _selectedAnswers = {}; // questionId -> selectedOptionId

  int _warningActive = 0;
  final int _warningTotal = 5;

  // Head pose detection state
  FaceMeshDetector? _faceMeshDetector;
  bool _processing = false;
  int _internalWarningCount = 0;

  // Calibration
  double _pitchOffset = 0.0;
  double _yawOffset = 0.0;
  bool _isCalibrated = false;
  final List<double> _calibrationYawSamples = [];
  final List<double> _calibrationPitchSamples = [];
  static const int _calibrationSampleCount = 15;

  // Debug/Preview values (temporary for testing)
  bool _faceDetected = false;
  String _direction = 'FORWARD';
  double _debugYaw = 0;
  double _debugPitch = 0;

  // Show calibration screen first
  bool _showCalibrationScreen = true;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadQuestions();
    _initHeadPoseDetection();
  }

  // ==================== HEAD POSE DETECTION LOGIC ====================

  void _initHeadPoseDetection() {
    _faceMeshDetector = FaceMeshDetector(
      option: FaceMeshDetectorOptions.faceMesh,
    );
  }

  Future<void> _processImage(CameraImage image) async {
    if (_processing || _faceMeshDetector == null) return;
    _processing = true;

    try {
      final inputImage = _toInputImage(image);
      if (inputImage == null) {
        _processing = false;
        return;
      }

      final meshes = await _faceMeshDetector!.processImage(inputImage);

      if (meshes.isNotEmpty) {
        final mesh = meshes.first;
        final points = mesh.points;

        if (points.length < 360) {
          _processing = false;
          return;
        }

        final noseTip = points[4];
        final foreHead = points[10];
        final chin = points[152];
        final leftEyeOuter = points[33];
        final rightEyeOuter = points[263];
        final leftEyeInner = points[133];
        final rightEyeInner = points[362];

        final eyeCenterX =
            (leftEyeOuter.x +
                rightEyeOuter.x +
                leftEyeInner.x +
                rightEyeInner.x) /
            4;

        final faceWidth = (rightEyeOuter.x - leftEyeOuter.x).abs();
        final faceHeight = (chin.y - foreHead.y).abs();

        if (faceWidth < 10 || faceHeight < 10) {
          _processing = false;
          return;
        }

        final noseOffsetX = noseTip.x - eyeCenterX;
        final rawYawRatio = noseOffsetX / faceWidth;

        final faceCenterY = (foreHead.y + chin.y) / 2;
        final noseOffsetY = noseTip.y - faceCenterY;
        final rawPitchRatio = noseOffsetY / faceHeight;

        // Auto-calibration
        if (!_isCalibrated) {
          _calibrationYawSamples.add(rawYawRatio);
          _calibrationPitchSamples.add(rawPitchRatio);

          if (_calibrationYawSamples.length >= _calibrationSampleCount) {
            _yawOffset =
                _calibrationYawSamples.reduce((a, b) => a + b) /
                _calibrationYawSamples.length;
            _pitchOffset =
                _calibrationPitchSamples.reduce((a, b) => a + b) /
                _calibrationPitchSamples.length;
            _isCalibrated = true;
            debugPrint(
              'Auto-calibrated - Yaw offset: $_yawOffset, Pitch offset: $_pitchOffset',
            );
          }

          if (mounted) {
            setState(() {
              _faceDetected = true;
              _debugYaw = rawYawRatio * 100;
              _debugPitch = rawPitchRatio * 100;
            });
          }

          _processing = false;
          await Future.delayed(const Duration(milliseconds: 100));
          return;
        }

        // Apply calibration offsets
        final yawRatio = rawYawRatio - _yawOffset;
        final pitchRatio = rawPitchRatio - _pitchOffset;

        // Store debug values
        _debugYaw = yawRatio * 100;
        _debugPitch = pitchRatio * 100;

        String horizontal = 'FORWARD';
        String vertical = 'FORWARD';

        // Thresholds
        if (yawRatio > 0.15) {
          horizontal = 'LEFT';
        } else if (yawRatio < -0.15) {
          horizontal = 'RIGHT';
        }

        if (pitchRatio > 0.12) {
          vertical = 'DOWN';
        } else if (pitchRatio < -0.12) {
          vertical = 'UP';
        }

        String newDirection;
        if (horizontal == 'FORWARD' && vertical == 'FORWARD') {
          newDirection = 'FORWARD';
        } else if (horizontal != 'FORWARD' && vertical != 'FORWARD') {
          newDirection = '$vertical-$horizontal';
        } else if (horizontal != 'FORWARD') {
          newDirection = horizontal;
        } else {
          newDirection = vertical;
        }

        // Only count warnings after calibration screen is dismissed
        if (!_showCalibrationScreen && newDirection != 'FORWARD') {
          _internalWarningCount++;
          int newDisplayedCount = _internalWarningCount ~/ 3;
          if (newDisplayedCount != _warningActive &&
              newDisplayedCount <= _warningTotal &&
              mounted) {
            setState(() {
              _warningActive = newDisplayedCount;
            });
          }
        }

        if (mounted) {
          setState(() {
            _faceDetected = true;
            _direction = newDirection;
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _faceDetected = false;
          });
        }
      }
    } catch (e) {
      debugPrint('Error processing image: $e');
    }

    await Future.delayed(const Duration(milliseconds: 300));
    _processing = false;
  }

  InputImage? _toInputImage(CameraImage image) {
    final camera = _cameraController?.description;
    if (camera == null) return null;

    final sensorOrientation = camera.sensorOrientation;
    InputImageRotation? rotation;

    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation = sensorOrientation;
      if (camera.lensDirection == CameraLensDirection.front) {
        rotationCompensation = (sensorOrientation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }

    if (rotation == null) return null;

    if (Platform.isAndroid) {
      final format = InputImageFormat.nv21;
      final allBytes = WriteBuffer();
      for (final plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      return InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: format,
          bytesPerRow: image.planes[0].bytesPerRow,
        ),
      );
    } else {
      final format = InputImageFormat.bgra8888;
      return InputImage.fromBytes(
        bytes: image.planes[0].bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: format,
          bytesPerRow: image.planes[0].bytesPerRow,
        ),
      );
    }
  }

  void _recalibrate() {
    setState(() {
      _isCalibrated = false;
      _calibrationYawSamples.clear();
      _calibrationPitchSamples.clear();
      _yawOffset = 0.0;
      _pitchOffset = 0.0;
      _showCalibrationScreen = true;
    });
  }

  void _startExam() {
    if (_isCalibrated) {
      setState(() {
        _showCalibrationScreen = false;
        _internalWarningCount = 0;
        _warningActive = 0;
      });
    }
  }

  // ==================== END HEAD POSE DETECTION LOGIC ====================

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      debugPrint('Camera permission denied');
      return;
    }

    WidgetsFlutterBinding.ensureInitialized();
    cameras = await availableCameras();
    final CameraDescription cam = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    _cameraController = CameraController(
      cam,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          Platform.isAndroid
              ? ImageFormatGroup.nv21
              : ImageFormatGroup.bgra8888,
    );
    await _cameraController!.initialize();

    // Start image stream for head pose detection
    await _cameraController!.startImageStream(_processImage);

    if (!mounted) return;
    setState(() => _cameraReady = true);
  }

  Future<void> _loadQuestions() async {
    try {
      final response = await ExamService.getExamQuestions(
        examId: widget.examId,
      );

      if (response['success'] == true) {
        setState(() {
          _examData = response['data'];
          _questions = _examData['questions'] ?? [];
          _isLoading = false;
          _errorMessage = null;
        });
      } else {
        setState(() {
          _isLoading = false;
          _errorMessage = response['message'] ?? 'Failed to load questions';
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Error loading questions: $e';
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _faceMeshDetector?.close();
    super.dispose();
  }

  void _handleAnswerSelection(int optionId) async {
    setState(() {
      _selectedAnswers[_questions[_currentQuestionIndex]['id']] = optionId;
    });

    // Save answer to API
    final response = await ExamService.saveAnswer(
      questionId: _questions[_currentQuestionIndex]['id'],
      selectedOptionIds: [optionId],
    );

    if (response['success'] != true) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(response['message'] ?? 'Failed to save answer'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _handleSubmitExam() async {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder:
          (context) => const Dialog(
            child: Padding(
              padding: EdgeInsets.all(20.0),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 20),
                  Text('Submitting exam...'),
                ],
              ),
            ),
          ),
    );

    try {
      final response = await ExamService.submitExam(examId: widget.examId);

      if (mounted) {
        Navigator.pop(context);

        if (response['success'] == true) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (_) => const QuizzesScreen()),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(response['message'] ?? 'Failed to submit exam'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _goToNextQuestion() {
    if (_currentQuestionIndex < _questions.length - 1) {
      setState(() => _currentQuestionIndex++);
    }
  }

  void _goToPreviousQuestion() {
    if (_currentQuestionIndex > 0) {
      setState(() => _currentQuestionIndex--);
    }
  }

  @override
  Widget build(BuildContext context) {
    // Show calibration screen first
    if (_showCalibrationScreen) {
      return _buildCalibrationScreen();
    }

    if (_isLoading) {
      return Scaffold(
        backgroundColor: Colors.white,
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        backgroundColor: Colors.white,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _loadQuestions,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_questions.isEmpty) {
      return Scaffold(
        backgroundColor: Colors.white,
        body: const Center(child: Text('No questions available')),
      );
    }

    final currentQuestion = _questions[_currentQuestionIndex];
    final List<dynamic> options = currentQuestion['options'] ?? [];
    final int? selectedOptionId = _selectedAnswers[currentQuestion['id']];
    final isLastQuestion = _currentQuestionIndex == _questions.length - 1;

    const Color brandRed = Color(0xFFE53935);
    const Color cardBorder = Color(0xFFE6E6E6);
    const double contentPadding = 16;

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Stack(
          children: [
            // Main content
            Padding(
              padding: const EdgeInsets.only(
                left: contentPadding,
                right: contentPadding,
                top: 13,
                bottom: contentPadding,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Top bar: Title + small warning points + question progress
                  Row(
                    children: [
                      // Title on the left
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _examData['examTitle'] ?? 'Exam',
                              style: const TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                                color: Color(0xFF002E8A),
                              ),
                            ),
                            const SizedBox(height: 10),
                            Row(
                              children: [
                                Icon(
                                  Icons.timer_outlined,
                                  color: Color(0xFF002E8A),
                                  size: 28,
                                ),
                                const SizedBox(width: 8),
                                Text(
                                  '${_examData['remainingMinutes'] ?? 0}:00',
                                  style: Theme.of(
                                    context,
                                  ).textTheme.headlineSmall?.copyWith(
                                    color: Color(0xFF002E8A),
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 56),
                          ],
                        ),
                      ),
                      const SizedBox(width: 12),
                      // Camera preview thumbnail (top-right)
                      if (_cameraController != null)
                        Container(
                          height: 125,
                          width: 155,
                          child: _CameraThumbnail(
                            cameraReady: _cameraReady,
                            controller: _cameraController!,
                            faceDetected: _faceDetected,
                            direction: _direction,
                          ),
                        ),
                    ],
                  ),

                  const SizedBox(height: 20),

                  // Full-width warning panel (matches visual emphasis)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: SizedBox(
                      width: MediaQuery.of(context).size.width * 0.57,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: const Color.fromARGB(255, 134, 171, 246),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: const Color(0xFF002E8A)),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: List.generate(_warningTotal, (i) {
                                final bool active = i < _warningActive;
                                return Padding(
                                  padding: EdgeInsets.only(
                                    right: i == _warningTotal - 1 ? 12 : 8,
                                  ),
                                  child: Icon(
                                    Icons.warning_amber_rounded,
                                    size: 26,
                                    color:
                                        active
                                            ? brandRed
                                            : Colors.grey.shade400,
                                  ),
                                );
                              }),
                            ),
                            const Divider(
                              color: Color(0xFF002E8A),
                              height: 20,
                              thickness: 2,
                            ),
                            const Text(
                              'Warning Points',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),

                  // DEBUG INFO - TEMPORARY (remove in production)
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.yellow.shade100,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.orange),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '🔧 DEBUG INFO (temporary)',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                            color: Colors.orange.shade800,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Face: ${_faceDetected ? "✅ Detected" : "❌ Not detected"}',
                          style: TextStyle(
                            fontSize: 11,
                            color: _faceDetected ? Colors.green : Colors.red,
                          ),
                        ),
                        Text(
                          'Direction: $_direction',
                          style: TextStyle(
                            fontSize: 11,
                            color:
                                _direction == 'FORWARD'
                                    ? Colors.green
                                    : Colors.red,
                          ),
                        ),
                        Text(
                          'Yaw: ${_debugYaw.toStringAsFixed(1)} | Pitch: ${_debugPitch.toStringAsFixed(1)}',
                          style: const TextStyle(fontSize: 11),
                        ),
                        Text(
                          'Internal warnings: $_internalWarningCount | Displayed: $_warningActive',
                          style: const TextStyle(fontSize: 11),
                        ),
                        InkWell(
                          onTap: _recalibrate,
                          child: Text(
                            '🔄 Tap to recalibrate',
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.blue,
                              decoration: TextDecoration.underline,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  // END DEBUG INFO
                  const SizedBox(height: 20),

                  // Question progress
                  Text(
                    'Question ${_currentQuestionIndex + 1}/${_questions.length}',
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF606060),
                    ),
                  ),

                  const SizedBox(height: 8),

                  // Question text
                  Text(
                    currentQuestion['text'] ?? 'No question text',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      height: 1.4,
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Options list
                  Expanded(
                    child: ListView.builder(
                      itemCount: options.length,
                      itemBuilder: (context, index) {
                        final option = options[index];
                        final optionId = option['id'];
                        final isSelected = selectedOptionId == optionId;
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: InkWell(
                            borderRadius: BorderRadius.circular(12),
                            onTap: () => _handleAnswerSelection(optionId),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 14,
                                vertical: 16,
                              ),
                              decoration: BoxDecoration(
                                color:
                                    isSelected
                                        ? const Color.fromARGB(
                                          255,
                                          134,
                                          171,
                                          246,
                                        )
                                        : Colors.white,
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(
                                  color:
                                      isSelected
                                          ? Color(0xFF002E8A)
                                          : cardBorder,
                                  width: isSelected ? 2 : 1,
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.04),
                                    blurRadius: 8,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: Row(
                                children: [
                                  Icon(
                                    isSelected
                                        ? Icons.radio_button_checked
                                        : Icons.radio_button_off,
                                    color:
                                        isSelected
                                            ? Color(0xFF002E8A)
                                            : Colors.grey,
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Text(
                                      option['text'] ?? 'No option text',
                                      style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                        color: Colors.black,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),

                  // Navigation buttons
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      if (_currentQuestionIndex > 0)
                        ElevatedButton(
                          onPressed: _goToPreviousQuestion,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.grey[300],
                            foregroundColor: Colors.black,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 22,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 0,
                          ),
                          child: const Text(
                            'Previous',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        )
                      else
                        const SizedBox.shrink(),
                      if (!isLastQuestion)
                        ElevatedButton(
                          onPressed: _goToNextQuestion,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF002E8A),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 22,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 0,
                          ),
                          child: const Text(
                            'Next',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        )
                      else
                        ElevatedButton(
                          onPressed: _handleSubmitExam,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Color(0xFF002E8A),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 22,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 0,
                          ),
                          child: const Text(
                            'Submit',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCalibrationScreen() {
    return Scaffold(
      backgroundColor: const Color(0xFF002E8A),
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 40),
            const Text(
              'Exam Setup',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Position yourself for proctoring',
              style: TextStyle(fontSize: 16, color: Colors.white70),
            ),
            const SizedBox(height: 40),

            // Camera preview
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color:
                          _isCalibrated
                              ? Colors.green
                              : (_faceDetected ? Colors.orange : Colors.red),
                      width: 4,
                    ),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child:
                      _cameraReady && _cameraController != null
                          ? CameraPreview(_cameraController!)
                          : const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Status info
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 24),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  // Face detection status
                  Row(
                    children: [
                      Icon(
                        _faceDetected
                            ? Icons.check_circle
                            : Icons.error_outline,
                        color: _faceDetected ? Colors.green : Colors.red,
                      ),
                      const SizedBox(width: 12),
                      Text(
                        _faceDetected ? 'Face Detected' : 'No Face Detected',
                        style: TextStyle(
                          color: _faceDetected ? Colors.green : Colors.red,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // Calibration status
                  Row(
                    children: [
                      Icon(
                        _isCalibrated
                            ? Icons.check_circle
                            : Icons.hourglass_empty,
                        color: _isCalibrated ? Colors.green : Colors.orange,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _isCalibrated
                                  ? 'Calibration Complete'
                                  : 'Calibrating...',
                              style: TextStyle(
                                color:
                                    _isCalibrated
                                        ? Colors.green
                                        : Colors.orange,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            if (!_isCalibrated)
                              Text(
                                'Look straight at the screen (${_calibrationYawSamples.length}/$_calibrationSampleCount)',
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),

                  // Debug values (temporary)
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.yellow.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          '🔧 Debug (temporary)',
                          style: TextStyle(
                            color: Colors.yellow,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Direction: $_direction',
                          style: TextStyle(
                            color:
                                _direction == 'FORWARD'
                                    ? Colors.green
                                    : Colors.red,
                            fontSize: 11,
                          ),
                        ),
                        Text(
                          'Yaw: ${_debugYaw.toStringAsFixed(1)} | Pitch: ${_debugPitch.toStringAsFixed(1)}',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // Start button
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _isCalibrated ? _startExam : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: const Color(0xFF002E8A),
                    disabledBackgroundColor: Colors.white.withOpacity(0.3),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: Text(
                    _isCalibrated ? 'Start Exam' : 'Please wait...',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Recalibrate button
            if (_isCalibrated)
              TextButton(
                onPressed: _recalibrate,
                child: const Text(
                  'Recalibrate',
                  style: TextStyle(color: Colors.white70),
                ),
              ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

// Separate widget for camera thumbnail to keep layout tidy
class _CameraThumbnail extends StatelessWidget {
  final bool cameraReady;
  final CameraController controller;
  final bool faceDetected;
  final String direction;

  const _CameraThumbnail({
    Key? key,
    required this.cameraReady,
    required this.controller,
    this.faceDetected = false,
    this.direction = 'FORWARD',
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Color borderColor = Colors.grey;
    if (faceDetected) {
      borderColor = direction == 'FORWARD' ? Colors.green : Colors.orange;
    } else {
      borderColor = Colors.red;
    }

    return Container(
      width: 155,
      height: 125,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderColor, width: 3),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child:
          cameraReady
              ? Stack(
                children: [
                  CameraPreview(controller),
                  // Small indicator
                  Positioned(
                    bottom: 4,
                    right: 4,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        direction,
                        style: TextStyle(
                          color:
                              direction == 'FORWARD'
                                  ? Colors.green
                                  : Colors.orange,
                          fontSize: 8,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                ],
              )
              : const Center(
                child: SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ),
    );
  }
}
