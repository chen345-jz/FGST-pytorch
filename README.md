# 姣背娉㈤浄杈剧偣浜戝姩浣?姝ユ€佽瘑鍒紙C++锛?
杩欐槸涓€涓彲鐩存帴钀藉湴鏀归€犵殑鏁版嵁澶勭悊涓庤瘑鍒伐绋嬶紝娴佺▼涓猴細

1. 鏁版嵁鍔犺浇锛坢anifest + 搴忓垪 CSV锛?2. 棰勫鐞嗭紙SNR/璺濈婊ゆ尝锛?3. 甯у唴 DBSCAN 鑱氱被
4. 鏃剁┖缁熻鐗瑰緛鎻愬彇
5. KNN 璁粌 / 璇勪及 / 鍗曟牱鏈娴?6. 妯″瀷淇濆瓨涓庡姞杞?7. 澶氬垎绫绘寚鏍囪緭鍑猴紙Accuracy / Precision / Recall / F1 / 娣锋穯鐭╅樀锛?
## 1. 鐩綍缁撴瀯

- `include/`锛氬ご鏂囦欢
- `src/`锛氬疄鐜?- `config/example.cfg`锛氱ず渚嬮厤缃?- `data/`锛氫綘鑷繁鐨勬暟鎹?- `model/`锛氳緭鍑烘ā鍨?
## 2. 鏁版嵁鏍煎紡

### 2.1 manifest

`data/manifest.csv` 鍒楀畾涔夛細

```csv
sequence_id,label,file_path
seq_0001,0,./data/sequences/seq_0001.csv
seq_0002,1,./data/sequences/seq_0002.csv
```

### 2.2 鍗曚釜搴忓垪

姣忎釜搴忓垪 CSV 鍒楀畾涔夛細

```csv
frame_idx,timestamp,x,y,z,doppler,snr
0,0,1.2,0.3,0.1,0.8,12.5
0,0,1.1,0.2,0.1,0.7,10.1
1,50,1.3,0.35,0.1,0.9,11.8
```

- `timestamp` 鍗曚綅榛樿姣
- 姣忚鏄竴涓偣
- `frame_idx` 鐩稿悓鍗冲睘浜庡悓涓€甯?
## 3. 鏋勫缓

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## 4. 杩愯

璁粌锛?
```powershell
./build/Release/mmwave_app.exe train ./config/example.cfg
```

璇勪及锛?
```powershell
./build/Release/mmwave_app.exe eval ./config/example.cfg
```

棰勬祴锛堝崟涓簭鍒楋級锛?
```powershell
./build/Release/mmwave_app.exe predict ./config/example.cfg ./data/sequences/seq_0001.csv
```

## 5. 鍜屼綘鐨勮鏂囧搴斿叧绯?
- 鐐逛簯棰勫鐞嗭細`src/preprocess.cpp`
- 鑱氱被锛歚Preprocessor::cluster_frame_dbscan`
- 鐗瑰緛宸ョ▼锛歚src/features.cpp`
- 璇嗗埆妯″瀷锛歚src/classifier.cpp`
- 鍙缁冨鍒嗙被妯″瀷锛堝姣斿熀绾匡級锛歚src/softmax_classifier.cpp`
- 璁粌/娴嬭瘯娴佺▼锛歚src/pipeline.cpp`
- 鎸囨爣涓庢贩娣嗙煩闃碉細`src/metrics.cpp`

## 6. 浣犻渶瑕佹敼鐨勬渶灏忛儴鍒?
濡傛灉浣犳暟鎹垪鍜岃繖閲屼笉涓€鑷达紝浼樺厛淇敼锛?
- `src/dataset.cpp`锛堣缁冮泦璇诲彇锛?- `src/pipeline.cpp` 閲岀殑 `load_one_sequence_csv`锛堥娴嬫牱鏈鍙栵級

鍏朵粬妯″潡鍙互鍏堜笉鍔ㄣ€?
## 7. 妯″瀷鍒囨崲锛堢敤浜庡姣斿疄楠岋級

鍦ㄩ厤缃腑浣跨敤锛?
```ini
classifier.type=knn
```

鎴栵細

```ini
classifier.type=softmax
softmax.learning_rate=0.05
softmax.epochs=200
softmax.l2=0.0001
```

`train/eval` 浼氳嚜鍔ㄨ緭鍑哄苟淇濆瓨鎸囨爣 CSV锛堢敱 `report.path` 鎸囧畾锛夈€?
## 8. PointNet++锛圕++锛?
鏈伐绋嬪凡棰勭暀 `classifier.type=fgst` 鎺ュ彛銆? 
瑕佺湡姝ｅ惎鐢?PointNet++ 璁粌/鎺ㄧ悊锛岄渶瑕?LibTorch锛?
```powershell
cmake -S . -B build -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH="浣犵殑/libtorch"
cmake --build build --config Release
```

褰撳墠浠撳簱宸叉彁渚?`fgst` 鐨?LibTorch 璁粌/鎺ㄧ悊璺緞锛堢畝鍖?PointNet++ 椋庢牸鍒嗗眰姹犲寲瀹炵幇锛夈€?鏈紑鍚?LibTorch 鏃讹紝`fgst` 妯″紡涓嶅彲鐢ㄣ€?
