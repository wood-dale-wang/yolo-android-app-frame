package org.example.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    context: Context,
    private val modelAssetPath: String = "yolov8n.tflite",
    private val labelsAssetPath: String = "labels.txt"
) {
    private val labels: List<String>
    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int

    init {
        val modelBuffer = loadModelBuffer(context, modelAssetPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)

        val inputShape = interpreter.getInputTensor(0).shape()
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        labels = loadLabels(context, labelsAssetPath)
    }

    suspend fun detect(
        bitmap: Bitmap,
        confidenceThreshold: Float = 0.3f,
        iouThreshold: Float = 0.45f,
        maxResults: Int = 200
    ): List<DetectionResult> = withContext(Dispatchers.Default) {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val input = preprocess(resized)

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputType = outputTensor.dataType()
        require(outputType == DataType.FLOAT32) {
            "Only FLOAT32 output tensors are currently supported."
        }

        val outputBuffer = ByteBuffer
            .allocateDirect(outputShape.reduce(Int::times) * 4)
            .order(ByteOrder.nativeOrder())

        interpreter.run(input.buffer, outputBuffer)
        outputBuffer.rewind()

        val values = FloatArray(outputShape.reduce(Int::times))
        outputBuffer.asFloatBuffer().get(values)

        val rawDetections = parsePredictions(
            values = values,
            outputShape = outputShape,
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
            confidenceThreshold = confidenceThreshold
        )

        nms(rawDetections, iouThreshold)
            .sortedByDescending { it.score }
            .take(maxResults)
    }

    fun close() {
        interpreter.close()
    }

    private fun preprocess(bitmap: Bitmap): TensorImage {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
        return imageProcessor.process(tensorImage)
    }

    private fun parsePredictions(
        values: FloatArray,
        outputShape: IntArray,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<DetectionResult> {
        if (outputShape.size != 3) {
            return emptyList()
        }

        val dim1 = outputShape[1]
        val dim2 = outputShape[2]

        val numBoxes: Int
        val channels: Int
        val transposed: Boolean

        if (dim1 > dim2) {
            numBoxes = dim1
            channels = dim2
            transposed = false
        } else {
            numBoxes = dim2
            channels = dim1
            transposed = true
        }

        if (channels < 5) {
            return emptyList()
        }

        val classCount = channels - 4
        val results = mutableListOf<DetectionResult>()

        for (i in 0 until numBoxes) {
            val cx = getOutput(values, transposed, channels, numBoxes, 0, i)
            val cy = getOutput(values, transposed, channels, numBoxes, 1, i)
            val w = getOutput(values, transposed, channels, numBoxes, 2, i)
            val h = getOutput(values, transposed, channels, numBoxes, 3, i)

            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until classCount) {
                val score = getOutput(values, transposed, channels, numBoxes, c + 4, i)
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            if (bestClass < 0 || bestScore < confidenceThreshold) {
                continue
            }

            val normalizedCoords = cx <= 2f && cy <= 2f && w <= 2f && h <= 2f

            val left = if (normalizedCoords) {
                (cx - (w / 2f)) * originalWidth
            } else {
                ((cx - (w / 2f)) / inputWidth) * originalWidth
            }.coerceIn(0f, originalWidth.toFloat())

            val top = if (normalizedCoords) {
                (cy - (h / 2f)) * originalHeight
            } else {
                ((cy - (h / 2f)) / inputHeight) * originalHeight
            }.coerceIn(0f, originalHeight.toFloat())

            val right = if (normalizedCoords) {
                (cx + (w / 2f)) * originalWidth
            } else {
                ((cx + (w / 2f)) / inputWidth) * originalWidth
            }.coerceIn(0f, originalWidth.toFloat())

            val bottom = if (normalizedCoords) {
                (cy + (h / 2f)) * originalHeight
            } else {
                ((cy + (h / 2f)) / inputHeight) * originalHeight
            }.coerceIn(0f, originalHeight.toFloat())

            if (right <= left || bottom <= top) {
                continue
            }

            results += DetectionResult(
                classId = bestClass,
                label = labels.getOrNull(bestClass) ?: "class_$bestClass",
                score = bestScore,
                boundingBox = RectF(left, top, right, bottom)
            )
        }

        return results
    }

    private fun getOutput(
        values: FloatArray,
        transposed: Boolean,
        channels: Int,
        numBoxes: Int,
        c: Int,
        i: Int
    ): Float {
        return if (transposed) {
            values[c * numBoxes + i]
        } else {
            values[i * channels + c]
        }
    }

    private fun nms(
        detections: List<DetectionResult>,
        iouThreshold: Float
    ): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val selected = mutableListOf<DetectionResult>()

        while (sorted.isNotEmpty()) {
            val candidate = sorted.removeAt(0)
            selected += candidate

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (candidate.classId == other.classId && iou(candidate.boundingBox, other.boundingBox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }

        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        if (right <= left || bottom <= top) {
            return 0f
        }

        val intersection = (right - left) * (bottom - top)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return intersection / (areaA + areaB - intersection)
    }

    private fun loadLabels(context: Context, path: String): List<String> {
        return try {
            context.assets.open(path).bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .toList()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun loadModelBuffer(context: Context, path: String): ByteBuffer {
        val bytes = context.assets.open(path).use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }
}
