import turtle
import time  #导入时间包
import random #随机包

import pygame							# 导入pygame资源包
file=r'4_2.mp3'		# 音乐的路径
pygame.mixer.init()						# 初始化
track = pygame.mixer.music.load(file)	# 加载音乐文件
pygame.mixer.music.play()				# 开始播放音乐流

#控制jerry上下左右跑
def up():
    jerry.setheading(90)
    jerry.forward(20)

def down():
    jerry.setheading(270)
    jerry.forward(20)

def left():
    jerry.setheading(180)
    jerry.forward(20)

def right():
    jerry.setheading(0)
    jerry.forward(20)


#设置游戏场景
playground = turtle.Screen()
#放入角色图片
playground.register_shape('tom.gif')
playground.register_shape('jerry.gif')
playground.onkeypress(up,'Up') #当上键被按下的时候
playground.onkeypress(down,'Down')
playground.onkeypress(left,'Left')
playground.onkeypress(right,'Right')

playground.listen()  #用来监听，当按键按下触发事件

writer = turtle.Turtle()
#颜色
writer.color('brown')
writer.hideturtle()
writer.penup()
writer.home()
writer.write("TOM & JERRY",align='center',font=("Comic Sans MS",50,'bold'))
writer.goto(0,-50)
writer.write("READY? 3,2,1,GO!",align='center',font=("Comic Sans MS",20,'bold'))
#屏幕沉睡3秒后开始游戏
time.sleep(3)

writer.clear()

tom = turtle.Turtle()
tom.shape('tom.gif')
tom.penup() #随机移动tom
tom.goto(random.randint(-200,200),random.randint(-200,200)) #x,y的值
tom.pendown()
tom.penup()
tom.pensize(3)
tom.color('blue')


jerry = turtle.Turtle()
jerry.shape('jerry.gif')
jerry.speed(0)
jerry.penup()
jerry.goto(random.randint(-200,200),random.randint(-200,200))
jerry.color('brown')

start = time.time() #开始时间
#tom自动追逐
while True:
    tom.setheading(tom.towards(jerry))
    tom.forward(5) #tom的速度，数值越大，速度越快
    if tom.distance(jerry) < 10: #当tom与jerry的距离为10足够小时，捉到jerry，游戏结束
        end = time.time() #结束时间
        playground.clear()
        jerry.goto(0,0) #jerry归位
        jerry.write("GAME OVER",align='center',font=("Comic Sans MS",50,'bold'))
        jerry.goto(0,-50)
        jerry.write("YOU SURVIVED {:.1f} SECONDS".format(end-start),align='center',font=("Comic Sans Ms",20,'bold'))
        tom.pu()
        tom.goto(-30,-80)
        tom.stamp() #tom图像印章
        jerry.pu()
        jerry.goto(90, -70)
        jerry.stamp()
        break


