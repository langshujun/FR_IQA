import torch

pth = {"四川":"成都", "湖南": "长沙"}
t = {"四川":"1", "湖南": "2"}
list =["四川", "湖南"]

for k,v in pth.items():
    if pth[k] == v:
        print('yes')

