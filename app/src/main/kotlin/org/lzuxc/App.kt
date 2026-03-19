package org.lzuxc

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import org.lzuxc.ui.MainViewModel
import java.io.File

@Composable
fun AppScreen(mainViewModel: MainViewModel = viewModel()) {
    val context = LocalContext.current
    val uiState = mainViewModel.uiState
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var pendingCaptureUri by remember { mutableStateOf<Uri?>(null) }
    var isCapturing by remember { mutableStateOf(false) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            runCatching { loadBitmapFromUri(context, uri) }
                .onSuccess { bitmap -> mainViewModel.detectFromBitmap(bitmap) }
                .onFailure {
                    mainViewModel.clearError()
                    scope.launch { snackbarHostState.showSnackbar("读取图片失败: ${it.message}") }
                }
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        isCapturing = false
        val captureUri = pendingCaptureUri
        pendingCaptureUri = null

        if (!success || captureUri == null) {
            scope.launch { snackbarHostState.showSnackbar("已取消拍照") }
            return@rememberLauncherForActivityResult
        }

        runCatching { loadBitmapFromUri(context, captureUri) }
            .onSuccess { bitmap -> mainViewModel.detectFromBitmap(bitmap) }
            .onFailure {
                mainViewModel.clearError()
                scope.launch { snackbarHostState.showSnackbar("读取拍照图片失败: ${it.message}") }
            }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) {
            launchCameraCapture(
                context = context,
                launcher = cameraLauncher,
                onUriReady = { uri ->
                    pendingCaptureUri = uri
                    isCapturing = true
                },
                onError = {
                    isCapturing = false
                    scope.launch { snackbarHostState.showSnackbar("准备拍照失败: ${it.message}") }
                }
            )
        } else {
            isCapturing = false
            scope.launch {
                val result = snackbarHostState.showSnackbar(
                    message = "未授予相机权限，无法拍照识别",
                    actionLabel = "去设置",
                    duration = SnackbarDuration.Long
                )
                if (result == SnackbarResult.ActionPerformed) {
                    openAppSettings(context)
                }
            }
        }
    }

    LaunchedEffect(uiState.errorMessage) {
        val message = uiState.errorMessage ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(message)
        mainViewModel.clearError()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.Start
    ) {
        Text(
            text = "YOLO ncnn 物体识别",
            style = MaterialTheme.typography.headlineSmall
        )
        Text(
            text = "支持本地文件和摄像头拍照，输出目标列表与标注图像。",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(top = 4.dp)
        )

        Spacer(modifier = Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(onClick = { galleryLauncher.launch("image/*") }) {
                Text("选择本地图片")
            }
            OutlinedButton(
                enabled = !uiState.isDetecting && !isCapturing,
                onClick = {
                    if (!hasCameraApp(context)) {
                        scope.launch { snackbarHostState.showSnackbar("未找到可用相机应用") }
                        return@OutlinedButton
                    }

                    if (hasCameraPermission(context)) {
                        launchCameraCapture(
                            context = context,
                            launcher = cameraLauncher,
                            onUriReady = { uri ->
                                pendingCaptureUri = uri
                                isCapturing = true
                            },
                            onError = {
                                isCapturing = false
                                scope.launch { snackbarHostState.showSnackbar("准备拍照失败: ${it.message}") }
                            }
                        )
                    } else {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            ) {
                Text("拍照识别")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (uiState.isDetecting) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        }

        val displayBitmap = uiState.annotatedImage ?: uiState.sourceImage
        if (displayBitmap != null) {
            Surface(
                tonalElevation = 1.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Image(
                    bitmap = displayBitmap.asImageBitmap(),
                    contentDescription = "Detection preview",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "识别结果 (${uiState.detections.size})",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )

        if (uiState.detections.isEmpty() && uiState.sourceImage != null && !uiState.isDetecting) {
            Text(
                text = "未检测到目标，请尝试更清晰的图片或降低阈值。",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(top = 6.dp)
            )
        } else {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                uiState.detections.forEach { item ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        shape = RoundedCornerShape(10.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        val box = item.boundingBox
                        Text(
                            text = "${item.label}  |  置信度 ${(item.score * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(start = 10.dp, end = 10.dp, top = 8.dp)
                        )
                        Text(
                            text = "坐标: [${box.left.toInt()}, ${box.top.toInt()}, ${box.right.toInt()}, ${box.bottom.toInt()}]",
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(start = 10.dp, end = 10.dp, top = 2.dp, bottom = 8.dp)
                        )
                    }
                }
            }
        }

        SnackbarHost(hostState = snackbarHostState)
    }
}

private fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        ImageDecoder.decodeBitmap(source)
    } else {
        @Suppress("DEPRECATION")
        MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
    }
}

private fun hasCameraPermission(context: Context): Boolean {
    return ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED
}

private fun hasCameraApp(context: Context): Boolean {
    val captureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
    return captureIntent.resolveActivity(context.packageManager) != null
}

private fun openAppSettings(context: Context) {
    val intent = Intent(
        Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
        Uri.fromParts("package", context.packageName, null)
    ).apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    context.startActivity(intent)
}

private fun launchCameraCapture(
    context: Context,
    launcher: ActivityResultLauncher<Uri>,
    onUriReady: (Uri) -> Unit,
    onError: (Throwable) -> Unit
) {
    runCatching {
        val imageFile = File.createTempFile(
            "capture_",
            ".jpg",
            context.cacheDir
        )
        FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            imageFile
        )
    }.onSuccess { uri ->
        onUriReady(uri)
        launcher.launch(uri)
    }.onFailure { error ->
        onError(error)
    }
}

