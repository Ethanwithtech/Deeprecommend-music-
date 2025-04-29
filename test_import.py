import sys
import os

print(f"Python 解释器路径: {sys.executable}")
print("Python 搜索路径:")
for path in sys.path:
    print(f"  - {path}")

try:
    import flask
    print(f"Flask 版本: {flask.__version__}")
    print("Flask 可以导入")
except ImportError as e:
    print(f"导入 Flask 失败: {e}")

try:
    import flask_cors
    print("flask_cors 可以导入")
    from flask_cors import CORS
    print("CORS 类可以导入")
except ImportError as e:
    print(f"导入 flask_cors 失败: {e}")

print("导入测试完成") 