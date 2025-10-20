import importlib.util
import sys, os
from write_stock_data import repair_data
from get_stock_data import get_data

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "zero"))
sys.path.append(os.path.join(base_dir, "one"))
sys.path.append(os.path.join(base_dir, "three"))
sys.path.append(os.path.join(base_dir, "seven"))

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def gen_train_data():
    # 当前路径
    base_dir = os.path.dirname(__file__)

    # 分别导入三个模块
    gen_zero = import_from_path("gen_zero_train_data", os.path.join(base_dir, "zero", "gen_train_data.py"))
    gen_three = import_from_path("gen_three_train_data", os.path.join(base_dir, "three", "gen_train_data.py"))
    gen_seven = import_from_path("gen_seven_train_data", os.path.join(base_dir, "seven", "gen_train_data.py"))
    gen_one = import_from_path("gen_one_train_data", os.path.join(base_dir, "one", "gen_train_data.py"))

    # 分别调用三个不同的函数
    print("开始生成zero model的训练数据...")
    gen_zero.gen_zero_train_data()
    print("生成zero model的训练数据结束")
    print("开始生成three model的训练数据...")
    gen_three.gen_three_train_data()
    print("生成three model的训练数据结束")
    print("开始生成seven model的训练数据...")
    gen_seven.gen_seven_train_data()
    print("生成seven model的训练数据结束")
    print("开始生成one model的训练数据...")
    gen_one.gen_one_train_data()
    print("生成one model的训练数据结束")

def train_model():
    # 当前路径
    base_dir = os.path.dirname(__file__)

    # 分别导入三个模块
    train_zero = import_from_path("train_zero_model", os.path.join(base_dir, "zero", "train.py"))
    train_three = import_from_path("train_three_model", os.path.join(base_dir, "three", "train.py"))
    train_seven = import_from_path("train_seven_model", os.path.join(base_dir, "seven", "train.py"))
    train_one = import_from_path("train_one_model", os.path.join(base_dir, "one", "train.py"))

    print("开始训练zero model...")
    train_zero.train_zero_model()
    print("zero model训练结束")
    print("开始训练three model...")
    train_three.train_three_model()
    print("three model训练结束")
    print("开始训练seven model...")
    train_seven.train_seven_model()
    print("seven model训练结束")
    print("开始训练one model...")
    train_one.train_one_model()
    print("one model训练结束")


def refresh_data():
    print("开始更新各交易所数据...")
    repair_data()
    print("各交易所数据更新结束")
    print("开始调出各交易所数据...")
    get_data()
    print("各交易所数据调出结束")

def refresh_model():
    # print("开始更新数据...")
    # refresh_data()
    print("生成训练数据...")
    gen_train_data()
    print("开始训练模型...")
    train_model()


if __name__ == '__main__':
    # refresh_model()
    # refresh_data()
    # get_data()
    refresh_model()
