from setuptools import setup, find_packages
import os

def load_requirements():
    """从 requirements.txt 读取依赖列表"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(requirements_path):
        raise FileNotFoundError("requirements.txt 文件不存在，请先创建")
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        # 过滤空行和注释行（以 # 开头的行）
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith("#")
        ]
    return requirements

setup(
    name="lm_cc",
    version="0.1.0",
    packages=["lm_cc"],          # 对应 scripts/lm_cc 目录
    package_dir={"lm_cc": "scripts/lm_cc"},
    install_requires=load_requirements(),
    python_requires=">=3.11",
    include_package_data=True,
)