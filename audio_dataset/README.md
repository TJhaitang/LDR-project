## 数据集构造计划
- 音频转文本
  - 使用tts生成语音数据
- 音频对话输入
  - 将对话数据的input使用tts转化为对话数据
- 音频描述
  - 使用clapcap对音频进行描述
- 音频+图像对话

## 模型选择
- tts
  - ali cosyvoice
- 音频描述
  - microsoft  clapcap
- 音频embedding
  - microsoft clap
    - 输出不含时间维度
  - ali paraformer
    - 目前来看表现不错，但不知道能不能找到embedding
  - meta wav2vec
    - 不支持中文似乎，在中文语音识别上表现糟糕
  - jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
    - 表现还可以，但是输出的seq_len有些长，12s的音频将占用600个token