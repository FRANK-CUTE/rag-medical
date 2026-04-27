### RAG医学问答系统
#### 1. 注意事项 
本项目数据请在https://ftp.ncbi.nlm.nih.gov/pub/pmc/ 获取，运行时放置在```/data```目录中
#### 2. 项目运行环境
```
GPU
RTX 5090(32GB) * 1
CPU
25 vCPU Intel(R) Xeon(R) Platinum 8470Q
内存
90GB
硬盘
系统盘:30 GB
数据盘:免费:50GB + 付费:100GB
```
#### 3. Ollama
本项目的大模型由Ollama工具提供，需要在官网下载最新版本，在运行环境安装，然后通过以下语句运行，
```
Ollama serve
```
