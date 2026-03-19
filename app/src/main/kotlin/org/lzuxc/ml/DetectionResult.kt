package org.lzuxc.ml

import android.graphics.RectF

data class DetectionResult(
    val classId: Int,
    val label: String,
    val score: Float,
    val boundingBox: RectF
)

