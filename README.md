![image](https://github.com/user-attachments/assets/f5a7ab22-b554-485f-bb51-61e1168e96e9)


# multi_agent_demo1
家装小能手——用多模态模型把大师的画装饰客厅
报告日期：2024年8月18日  
项目负责人：Rey Cui   

#### 一、项目概述：

本项目旨在开发一个多模态数据分析Agent，该Agent能够接收艺术家的画作作为输入，通过分析提取画作的关键元素，并将其转换为文字描述。随后，基于编辑后的文字描述生成一张适合客厅装饰的新图片。这种技术可以广泛应用于家居装饰、艺术品再创作等领域，帮助用户快速定制个性化的室内装饰画。

#### 二、技术方案与实施步骤

2.1 模型选择  

- 用 microsoft/phi-3-vision 大模型来进行图片分析, 得到具体的文字描述
- 用 llama3.1 大模型进行文字的再创造, 精准修改，并添加个性化信息
- 用 stable-diffusion-3-medium 来生成最终的图片


2.2、数据的构建： 

整个过程存在图片-->base64编码-->图片的过程。首先需要将图片转换为base64编码，然后通过API接口发送给 microsoft/phi-3-vision 模型，模型返回文字描述。stable-diffusion-3-medium 模型生成的图片也是base64编码，最终需要再次转换为图片格式，方便终端的显示。


```python
from io import BytesIO 
from PIL import Image
import base64

# 将图片转成base64编码
def image2b64(image_file):
    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
        return image_b64

# base64格式转图片
def b64_to_image(x):
    base64_str = x.get("image")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image
```

#### 三、实施步骤：

3.1 环境搭建：

3.1.1 安装Anaconda或Miniconda:  
- 如果您尚未安装Anaconda或Miniconda，可以从其官方网站下载并安装。
- 如果您已经有了Python环境，并且不需要使用Anaconda或Miniconda的特性，您可以跳过此步骤。
3.1.2 创建Python 3.8虚拟环境:  
- 打开Anaconda Prompt。
- 运行以下命令创建一个新的Python 3.8环境：
```bash
conda create --name ai_endpoint python=3.8
```
3.1.3 激活环境:
- 使用以下命令激活新环境：
```bash
conda activate ai_endpoint 
```
3.1.4 安装必要的包:
- 安装所需的Python包：
```python
# 创建python 3.8虚拟环境
conda create --name ai_endpoint python=3.8
# 进入虚拟环境
conda activate ai_endpoint
# 安装nvidia_ai_endpoint工具
pip install langchain-nvidia-ai-endpoints
# 安装Jupyter Lab
pip install jupyterlab
# 安装langchain_core
pip install langchain_core
# 安装langchain
pip install langchain
pip install -U langchain-community
# 安装matplotlib
pip install matplotlib
# 安装Numpy
pip install numpy
# 安装faiss, 这里安装CPU版本
pip install faiss-cpu==1.7.4
# 安装OPENAI库
pip install openai
# 安装gradio
pip install gradio
```

3.2. 代码实现： 

![image](https://github.com/user-attachments/assets/0b77306d-6289-4147-9c64-958b6f161a50)

![image](https://github.com/user-attachments/assets/c6ec5624-b064-4e6d-993a-c2bdd18f9341)

![image](https://github.com/user-attachments/assets/ebb5484e-ef21-407c-8582-66a774010665)

#### 四、项目成果与展示：

![image](https://github.com/user-attachments/assets/13cd6e57-469c-448a-8654-65842a2f17d3)


#### 五、问题与解决方案：

Q1：整个过程很消耗 NIM 的 credits;
A1：多注册几个账号，或者单步测试，尽量减少credits的消耗。

Q2：图片与base64编码转换的效率；
A2：由于代码不熟练，这里卡了很久，需要更多细心。

Q3：整个过程，图片的size没有做控制；
A3：需要后续继续完善。

Q4：图片推理过程较慢，生成的图片不可控；
A4：meta/llama-3.1-405b-instruct，stable-diffusion-3-medium 的推理过程较慢，需要耐心等待。可以尝试其他模型，或者优化prompt。


#### 六、项目总结与展望：

整个项目来源于灵感一现：在熟悉 microsoft/phi-3-vision-128k-instruct 模型的过程中，随机输入了一张梵高的《向日葵》，意外在stable-diffusion-3-medium 模型生成了很棒的图片。phi-3-vision 模型可以提取出图片中的关键元素，并生成文字描述，而 stable-diffusion-3-medium 模型则可以根据文字描述生成图片，正好完善了这个项目。这个项目的实现过程中，也遇到了很多问题，比如图片与base64编码转换的效率，图片推理过程较慢，生成的图片不可控等。但是，通过不断优化和调试，最终实现了一个能够高效、准确、高效的多模态数据分析Agent。


#### 七、特别感谢

特别感谢 NVIDIA AI 训练营的各位老师和同学，为项目提供了 invaluable 的帮助。[nvidia.cn/training/online/](https://www.nvidia.cn/training/online/)

#### 八、附件与参考资料
 
环境安装参考：https://blog.csdn.net/kunhe0512/article/details/140910139  
