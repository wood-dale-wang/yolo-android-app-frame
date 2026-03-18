package org.example.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import org.example.ml.DetectionResult
import org.example.ml.ImageAnnotator
import org.example.ml.YoloDetector

data class AppUiState(
    val sourceImage: Bitmap? = null,
    val annotatedImage: Bitmap? = null,
    val detections: List<DetectionResult> = emptyList(),
    val isDetecting: Boolean = false,
    val errorMessage: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val detector = YoloDetector(application)

    var uiState by mutableStateOf(AppUiState())
        private set

    fun clearError() {
        uiState = uiState.copy(errorMessage = null)
    }

    fun detectFromBitmap(bitmap: Bitmap) {
        val normalizedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        uiState = uiState.copy(
            sourceImage = normalizedBitmap,
            annotatedImage = null,
            detections = emptyList(),
            isDetecting = true,
            errorMessage = null
        )

        viewModelScope.launch {
            try {
                val detections = detector.detect(normalizedBitmap)
                val annotated = ImageAnnotator.annotate(normalizedBitmap, detections)
                uiState = uiState.copy(
                    annotatedImage = annotated,
                    detections = detections,
                    isDetecting = false,
                    errorMessage = null
                )
            } catch (e: Exception) {
                uiState = uiState.copy(
                    annotatedImage = null,
                    detections = emptyList(),
                    isDetecting = false,
                    errorMessage = e.message ?: "Detection failed."
                )
            }
        }
    }

    override fun onCleared() {
        detector.close()
        super.onCleared()
    }
}
