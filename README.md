# ekf-slam
拡張カルマンフィルタをベースとしたSLAMのコードです．

[Python Robotics](https://github.com/AtsushiSakai/PythonRobotics)のEKF SLAMのコードを参考にしています．

また，実装したアルゴリズムの解説はekf_slam.pdfに記載しています．

# Requirement
cmake

Eigen 3

# build
```
 $ mkdir build
 $ cd build
 $ cmake ../
 $ make -j 8
```
実行ファイルは `build/bin` に生成されます．

