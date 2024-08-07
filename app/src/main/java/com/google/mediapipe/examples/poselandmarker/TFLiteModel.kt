package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.Locale
import kotlin.math.sqrt

class TFLiteModel(private val context: Context) : TextToSpeech.OnInitListener {

    private lateinit var interpreter: Interpreter
    private lateinit var inputBuffer: TensorBuffer
    private lateinit var outputBuffer: TensorBuffer
    private lateinit var inputShape: IntArray
    private lateinit var outputShape: IntArray
    private var textToSpeech: TextToSpeech? = null
    private var isSpeaking = false
    private val handler = Handler(Looper.getMainLooper())

    private val landmarkNames = listOf(
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky_1", "right_pinky_1",
        "left_index_1", "right_index_1", "left_thumb_2", "right_thumb_2",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    )

    init {
        loadModel()
        prepareBuffers()
        textToSpeech = TextToSpeech(context, this)
    }

    private fun loadModel() {
        val modelFile = FileUtil.loadMappedFile(context, "cnn_lite_model.tflite")
        interpreter = Interpreter(modelFile)
    }

    private fun prepareBuffers() {
        inputShape = interpreter.getInputTensor(0).shape()
        outputShape = interpreter.getOutputTensor(0).shape()

        inputBuffer = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
    }

    fun classify(poseLandmarkerResult: PoseLandmarkerResult): String {
        if (poseLandmarkerResult.landmarks().isEmpty()) {
            return "Pose tidak terdeteksi"
        }

        else {
            val inputData = processInput(poseLandmarkerResult)
            inputBuffer.loadArray(inputData)

            interpreter.run(inputBuffer.buffer, outputBuffer.buffer.rewind())

            val outputData = outputBuffer.floatArray
            return interpretOutput(outputData)
        }
    }

    private fun processInput(poseLandmarkerResult: PoseLandmarkerResult): FloatArray {

        val landmarks = poseLandmarkerResult.landmarks().flatten()
        val centerX = (landmarks[landmarkNames.indexOf("right_hip")].x() + landmarks[landmarkNames.indexOf("left_hip")].x()) * 0.5f
        val centerY = (landmarks[landmarkNames.indexOf("right_hip")].y() + landmarks[landmarkNames.indexOf("left_hip")].y()) * 0.5f

        val shouldersX = (landmarks[landmarkNames.indexOf("right_shoulder")].x() + landmarks[landmarkNames.indexOf("left_shoulder")].x()) * 0.5f
        val shouldersY = (landmarks[landmarkNames.indexOf("right_shoulder")].y() + landmarks[landmarkNames.indexOf("left_shoulder")].y()) * 0.5f

        var maxDistance = 0f
        for (landmark in landmarks) {
            val distance = sqrt((landmark.x() - centerX) * (landmark.x() - centerX) + (landmark.y() - centerY) * (landmark.y() - centerY))
            if (distance > maxDistance) {
                maxDistance = distance
            }
        }
        val torsoSize = sqrt((shouldersX - centerX) * (shouldersX - centerX) + (shouldersY - centerY) * (shouldersY - centerY))
        maxDistance = maxOf(torsoSize * 2.5f, maxDistance)

        val input = FloatArray(132)
        for (i in landmarks.indices) {
            val landmark = landmarks[i]
            input[i * 4] = (landmark.x() - centerX) / maxDistance
            input[i * 4 + 1] = (landmark.y() - centerY) / maxDistance
            input[i * 4 + 2] = landmark.z() / maxDistance
            input[i * 4 + 3] = 1.0f
        }

        for (i in input.indices step 12) {
            println("Input Row ${i / 12}: ${input.copyOfRange(i, i + 12).joinToString(", ")}")
        }

        return input
    }

    private fun interpretOutput(outputData: FloatArray): String {
        val maxIndex = outputData.indices.maxByOrNull { outputData[it] } ?: -1

        println("Class $maxIndex with confidence ${outputData[maxIndex]}")

        val resultText = when (maxIndex) {
            0 -> "Normal"
            1 -> "Kaki Di Atas"
            2 -> "Tidur"
            else -> "Error"
        }

        if (!isSpeaking) {
            isSpeaking = true
            handler.postDelayed({
                textToSpeech?.speak(resultText, TextToSpeech.QUEUE_FLUSH, null, null)
                isSpeaking = false
            }, 2000) // 2 seconds delay
        }

        return resultText
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech?.language = Locale("id", "ID")
        }
    }

    fun shutdown() {
        textToSpeech?.shutdown()
    }
}
