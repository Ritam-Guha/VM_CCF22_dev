import os, sys
import os.path as path
abs_path_pkg = path.abspath(path.join(__file__ ,"../"))
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path_pkg)