# 使用 Miniconda3 作为基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 解压 Conda 环境
RUN mkdir -p /opt/conda/envs/myenv && \
    tar -xzf /app/myenv.tar.gz -C /opt/conda/envs/myenv && \
    rm /app/myenv.tar.gz

RUN conda run -n myenv pip install --no-cache-dir -r /app/requirements.txt && \
    rm /app/requirements.txt \

# 设置默认 shell 以激活 Conda 环境
SHELL ["/bin/bash", "-c"]

# 运行 Python 应用
CMD ["bash", "-c", "source activate myenv && python api.py"]