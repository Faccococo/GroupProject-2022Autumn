#TODO:那个cpp的改造后的方法的输出格式应为"X Y Z"形式（不包括双引号）
#把编译后的.exe文件放在和现在这个main.py文件同级目录下!（就是替换掉现在这个test.exe文件）
import subprocess
import os

def Locator_vicon():
    os.environ['path'] += ';.\\'
    # print(os.environ['path'])
    #下面这个Popen参数中的第一项为那个.exe文件的名字
    p = subprocess.Popen(['vrpn_print_devices.exe', 'wand@192.168.0.199'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    m=stdout.decode()
    n=m.split("\n")[1]
    n = n.split(" ")
    # #x,y,z即为眼睛所在点的三维坐标
    x=n[0]
    y=n[1]
    z=n[2]
    return x, y, z
# print(m)
# print("1 2 3".split(" "))
# print(stderr.decode('gbk'))
