package org.lzuxc.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.max

object ImageAnnotator {
    fun annotate(
        source: Bitmap,
        detections: List<DetectionResult>
    ): Bitmap {
        val mutable = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        val stroke = max(2f, source.width / 320f)
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = stroke
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            textSize = max(24f, source.width / 28f)
            isAntiAlias = true
        }
        val labelBgPaint = Paint().apply {
            color = Color.argb(170, 255, 0, 0)
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        detections.forEach { detection ->
            val rect = detection.boundingBox
            canvas.drawRect(rect, boxPaint)

            val text = "${detection.label} ${(detection.score * 100).toInt()}%"
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize
            val left = rect.left.coerceAtLeast(0f)
            val top = (rect.top - textHeight - 10f).coerceAtLeast(0f)
            val right = (left + textWidth + 14f).coerceAtMost(source.width.toFloat())
            val bottom = (top + textHeight + 8f).coerceAtMost(source.height.toFloat())

            canvas.drawRect(left, top, right, bottom, labelBgPaint)
            canvas.drawText(text, left + 7f, bottom - 6f, textPaint)
        }

        return mutable
    }
}

