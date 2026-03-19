# YOLO物体识别app框架

## 技术概要

- API版本：minSDK 24
- 显示：原生Jetpack Compose
- 模型运行框架：ncnn

## 安装与运行

### android开发环境

首先你需要一个android开发环境，可以使用android studio，比较简单，略过。

如果你使用vscode作为主要开发环境，那么你需要单独配置环境，这里推荐博文[【简书】不想用Android Studio，教你如何用vscode搭建原生app开发环境](https://www.jianshu.com/p/378930364493)。

提示：
- 建议添加环境变量`ANDROID_HOME`指向SDK根目录，并使用这个环境变量为基础构建其他环境变量。
- 你大概率需要`JDK17`而不是更高版本，似乎会有兼容错误。
- 博文提到的配置以及配置好了，见`.vscode`文件夹
- 模拟器调试我建议使用mumu模拟器，而不是自带的`emulator`（mumu需要自己开adb调试开关）。
- gradle国内使用大概率会碰到网络错误，有魔法的可以开魔法，也可以配置国内镜像站。
- 以及我不清楚这个项目的gradle配置是不是硬编码了本机安装的gradle，是的话就改一下就好了。

### python环境

最主要的包就是`ultralytics`，其他包会自动下载，所以我这里也没有给出环境包列表。

### ncnn环境

ncnn我是直接把so库放在项目中的，版本是`ncnn-20260113-android-vulkan-shared`，好像需要自己下载zip放置在根目录，因为编译时会从压缩包提取文件。

### 模型

建议新建`models`文件夹，将你下载的yolo模型放在其中。模型转换使用`tools/export_yolo_tflite.py`，具体命令在`doc/yolo-detection-api.md`中有。

`app/src/main/assets/model_config.properties`配置具体使用哪个模型。