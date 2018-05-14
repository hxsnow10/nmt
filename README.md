Nmt
===============
Now only code for nmt based sent representation.

* baidu_translate   百度翻译的接口代码
* nmt               google nmt code, almost not changed, 
                        http://wiki.baifendian.com/pages/viewpage.action?pageId=18162692
* sents_rep         google nmt based multilingual sent representation (main work)

## 核心思想
多语的句子表示，核心在于怎么利用多语对齐信息。
常用的方法包括bilingual nmt, multiligual nmt, etc.
具体地，
1) 双语，独立训练en->de, de->en的翻译，把他们中间表示拿出来，连接作为任意句子的句子表示。实际上可能需要en->de->en。
即我这里sents_rep的思路。不方便能多语。
2) 双语(多语), 每个语言配一个encode与一个decoder，期望对齐文本encoder表示的distance尽量小。
目前没有实现，有相关论文。应该也好实现。(更推荐思路)
3) 还有一些不需要decoder的思路，即encoder表示尽量靠近。

所有这些表示可以与监督任务解耦合；可以通过分类任务评估效果；可以嵌入在其他任务里提升单语的效果(数据的迁移)。

sents_rep 期望使用之前训练好的2个nmt，利用1的思路，来做情感分析。
没有跑出结果&评估，代码应该是可以跑的，主要是需要提前训练出比较好的nmt模型。
