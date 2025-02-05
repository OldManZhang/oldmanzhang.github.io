---
title: seq2seq 手工实现及原理分析
date: 2024-08-13 10:18:38
tags: 
    - 学肥AI
    - seq2seq
    - 深度学习
categories: 学肥AI
description: "现实中，有一类问题是 输入输出不定长 的，比如 
1. 翻译，从中文到英文
2. 文生图，一段话生成一个图片
3. 摘要，总结一段话的信息
所以 seq2seq 就是为了解决这种 一串序列 生成 另外一串序列 问题的模型。"
cover: https://qiniu.oldzhangtech.com/cover/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3%202.jpg
---


## 背景问题
现实中，有一类问题是 **输入输出不定长** 的，比如

1. 翻译，从中文到英文
2. 文生图，一段话生成一个图片
3. 摘要，总结一段话的信息

所以 `seq2seq` 就是为了解决这种 一串序列 生成 另外一串序列 问题的模型。
## 原理
`seq2seq`，`sequence to sequence`，也有另外一种叫法 `encoder and decoder`。他是一种上层模型架构，即是组合模型，他可以由不同的底层模型来实现。

我们可以先看原理图。
_原理图_  
![](https://qiniu.oldzhangtech.com/mdpic/65e94f22-668b-4d3f-accc-57617096e7d4_cd0a100f-9d6f-44db-86c6-5061e8e974ae.jpeg)
从原理图中可以知道，`seq2seq `模型 有以下的特征：

1. 模型都会有一个 `Encoder` ，一个 `Decoder`，和一个 `Context`
2. `Encoder` 就是字面意思的 -- 编码器，`src_input` 经过`Encoder` 处理，输出 `Context` 中
3. 同理，`Decoder` 就是解码器，`tgt_input` 和 `Context` 经过 `Decoder` 处理, 输出 `tgt_output`
4. `Encoder` 和 `Decoder` 都必须能够识别 `Context`
> src： source， tgt： target


🔥 `Context` 的组成是非常重要的，他是 `Encoder` 和 `Decoder` 是能够识别的一个介质，是链接两者的桥梁。这种介质可以是 _隐状态_，可以是 _注意力的加权计算值_，等等，这些都由底层的模型来决定的。

就好比国际贸易中，我们想买澳大利亚铁矿。 美元是硬通货，中间介质，ZG 和 土澳 都认美元，所以 ZG encoder 先把 RMB 转成 Dollar，给到土澳 decoder，土澳再换回自己的 澳元。

🔥 **不定长**，输入值（比如，长度是 8）在 `Encoder` 都转换成统一的 `Context`（比如，128 X 512 的 2 层神经网络），同时 输出值的长度（比如，长度是 10 ） 由 `Decoder` 和 `Context`  来决定，已经与输入值无关了。

同时，`seq2seq` 仅仅是上层架构，底层实现的模型是啥都可以视情况而定。比如，底层可以是 `RNN`，可以是 `LSTM`，也可以是 `GRU`， 也可以是 `Transformer`。本文例子中使用 `RNN` 来实现。
## 例子 -- 翻译
> 下面是手工实现一个基于 `RNN` 的 `seq2seq` 模型。可运行的 ipynb 文件的[链接](https://gitee.com/oldmanzhang/resource_machine_learning/blob/master/deep_learning/seq2seq.ipynb)。

### 任务目标   
例子的目标，从有限的翻译资料中，训练出翻译的逻辑，实现从英文翻译成法文。

### 分析任务  
> 这里先不讨论字符的处理流程（清洗字符，过滤特殊字符等），所有的流程简单化，仅仅是验证模型的使用。

1. 翻译是一个“分类”任务
2. 这个是一个不定长的输入和输出的，所以使用 `seq2seq` 的模型
3. 同时输入和输出是有时间序列的，所以底层模型使用带有记忆能力的模型，我们使用 `RNN`

❓ 为什么是一份分类的任务？
这其实是 `word2index` 的过程，每个 `word` 就是一个分类。举例：比如 输入的是英文，英文中的一共有 4000 个单词，那么输入的分类就是 4000 ；输出的是法文，法文中的一共有 2000 个单词，那么输出的分类就是 2000。

### 代码结构
![](https://qiniu.oldzhangtech.com/mdpic/efa06cc1-1371-4e15-a0fd-0e5505739279_54099b97-f43c-47fc-ab0a-7f855bc6ad50.jpeg)
上图是 数据在 seq2seq 流动中串起不同组件的过程。


_组件说明：_   

1. `word_index`，就是把单词转换成 `index`
2. `embedding`，就要把离散的 `index` 转换成可以计算的连续的 `embedding`，适合模型的计算
3. `word_index` 和 `embedding` 正常情况是 输入和输出都不能共用的
4. `encoder` 里面有 `embedding`，`rnn`
   1. `rnn` 输入 `src`， 输出 `hidden` 隐状态，即 `Context`
5. `decoder` 里面有 `embedding`，`rnn`，`full_connect`
   1. `rnn `**循环**叠加输入 `tgt_input` 和 `Context`， 输出 `new hidden`,  `tgt_output`
   2. `full_connect` 负责把 `tgt_output` 生成真正的 `real_tgt_ouput`
> 了解他们的具体职责后再去看他们的代码就清晰多了


_代码片段分析_  
```python
# Define the Encoder RNN
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, input_seq, hidden):
        # 内部进行 embedding
        # 传入的是 input_indices
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
```
上面是 `encoder` 的代码，作用就是：

1. `src_input` 转成 `embedding`
2. `rnn` 把 `embedding` 转成 `hidden`，即 `Context`


```python
# Define the Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, ).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden):
        # 内部进行 embedding
        # 传入的是 input_indices
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        # 就是 全链接层 从 hidden -》 output_feature
        output = self.out(output.squeeze(1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
```
上面是 `dncoder` 的代码，与 `encoder `比较多了一个 `full connect` 使用

1. `tgt_input` 转成 `embedding`
2. `rnn` 把 `embedding` 转成 `hidden` 和 `output`
3. `full conect `再把 `output` 转成 `output_feature`

```python
# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # [1]
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_seq, tgt_seq, teacher_forcing_ratio=0.5):
        batch_size = src_seq.size(0)
        # count of words [[0, 1, 2, 9]]
        max_len = tgt_seq.size(1)
        # 11
        tgt_vocab_size = self.decoder.out.out_features
        # 有 11 个隐状态，就是 target 中的唯一值
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size)
        
        encoder_hidden = self.encoder.init_hidden()
        # encoder 的作用是 输出 hidden， output 就没有什么意义了
        # [2]
        encoder_output, encoder_hidden = self.encoder(src_seq, encoder_hidden)
        
        # tgt_seq 作用，就是取得第一个 <sos> token
        decoder_input = tgt_seq[:, 0].unsqueeze(1)  # Start with <sos>
        decoder_hidden = encoder_hidden
        
        # tgt_seq 作用，截取输出的长度
        # 不取 0，是因为 “0“ index 是一个 <sos>
        # [3]
        for t in range(1, max_len):
            # [4]
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # decoder_output shape (1,11)，其实是一个多分类的问题
            # 与 outputs[:, t] = decoder_output 是一样的，因为 batch_size 恒等于 1，所以暂时影响不大，但是实际应用中，应该要改成对应的 batch
            outputs[:, t, :] = decoder_output
            top1 = decoder_output.argmax(1).unsqueeze(1)
            # 这里是取巧了，teacher_forcing_ratio 是取巧了。
            # decode_input_t+1 有时是 decode_output_t， 有时是 real_target_seq_t
            # [5]
            decoder_input = tgt_seq[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1
        
        return outputs
```
上面的代码是 `seq2seq` 模型的定义。

_训练过程_   
可以检查数据在这个模型中流动如下：

1. [1] 里面包含了一个 `encoder` 和 `decoder`
2. [2] `forword` 时,  `encoder` 转换 `src_input` 成 `hidden`
3. [3] 开始 `decoder` 循环，最大长度是 `max_len`。初始化即是： `decoder_input = “<sos> index“`，`decoder_hidden = encoder_hidden`
4. [4] `decoder` 输出是 `output_index` + `new_hidden`
5. [5] `decoder_input+= output_index`, `decoder_hidden += new_hidden` 叠加后再走步骤 [3] 循环

💡 `teacher_forcing` 是什么？
就是训练的时候，有一定的概率输出是 _真实值_ 而不是 _预测值_。就能是模型更加快的收敛，加速模型的学习。但是过于依赖 _真实值_，就会导致泛化能力差。`teacher_forcing_ratio` 就可以调整阈值。

_推理过程_  
```python
        input_seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        # 全部的input，都转成 hidden
        encoder_hidden = model.encoder.init_hidden()
        # encoder 和 decoder 的使用是分开的、
        # [1]
        encoder_output, encoder_hidden = model.encoder(input_seq, encoder_hidden)
        # [2]
        decoder_input = torch.tensor([[fra_word2idx['<sos>']]], dtype=torch.long)  # Start token
        decoder_hidden = encoder_hidden
        
        translated_sentence = []

        # [3]
        for _ in range(max_length):
            # decoder_input 是逐步的累加的，就是 word1+word2+word3...
            # 第一个 decoder_hidden 是 encoder_hidden
            # 从第二个开始，就是循环得到 decoder_hidden 不停的传入
            # encoder 和 decoder 的使用是分开的
            # [4]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            top1 = decoder_output.argmax(1).item()
            # "<UNK>", which stands for “unknown.”
            # [5]
            translated_word = fra_idx2word.get(top1, "<UNK>")
            translated_sentence.append(translated_word)

            # [6]
            if translated_word == '<eos>':  # End of sentence
                break
            
            decoder_input = torch.tensor([[top1]], dtype=torch.long)  # Next input token
        
        return translated_sentence
```
_推理过程_ 和 _训练过程_，具体原理一致。 有以下的差异点需要注意：

1. 如何定义开始输出的标志
2. 如何定义结束输出的标志
3. 如何定义不认识字符的标志

代码分析：

1. [1] 单独使用 `seq2seq's encoder`，且 _一次性_ 生成 `encoder hidden`
2. [2] `decoder_input` 初始化，以  '<sos>' 开头，标志开始输出
3. [3] `decoder` 开始循环
   1. [4] 单独使用 `seq2seq's decoder`, 输出 `ouput `和 `new_hidden`
   2. [5] 碰到不认识的分类，就使用 '<UNK>'取代
   3. [6] 如果遇到 '<eos>' 字符就直接结束循环
   4. 回到 [3] 继续循环

### 结果
```python
# 训练结果
Epoch: 0, Loss: 2.820215034484863
Epoch: 100, Loss: 1.0663029670715332
Epoch: 200, Loss: 1.1840879678726197
Epoch: 300, Loss: 1.224123215675354
Epoch: 400, Loss: 1.0645174384117126
Epoch: 500, Loss: 1.061875820159912
Epoch: 600, Loss: 1.0744179487228394
Epoch: 700, Loss: 1.0767890691757203
Epoch: 800, Loss: 1.099305510520935
Epoch: 900, Loss: 1.1019723176956178


# 预测
test_sentence = ["i", "am"]
translation = translate(model, test_sentence)
print("Translation:", " ".join(translation))

'''
Translation: nous sommes <eos>
'''
```
## 总结

1. `seq2seq` 是一种上层模型架构，应对输入和输出**不定长**的场景
2. `seq2seq` 底层可以由**不同**的模型构成
3. `seq2seq` 的 `Context` 是保存了**上下文信息**，是 `encoder` 和 `decoder` 都必须能识别的格式
## 

