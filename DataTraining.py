# ----------------导入相关package-----------------
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch.utils.data as Data


if __name__ == '__main__':

    torch.manual_seed(1)    # reproducible

    ## --------------------载入数据---------------------

    # 定义数据量多少
    data_size = 10000

    ## ----------------1.位置信息--------------

    ## 将数据转化成载入并转化成numpy
    #data_transformation = np.loadtxt('tf.txt',dtype= 'str',skiprows=0,delimiter=",")


    ##只筛选特定的行
    ##data_transformation_extratct = data_transformation[1:20, [0,5,6,7,8,9,10,11]]
    #data_transformation_extratct = data_transformation[ : , [0,5,6,7,8,9,10,11]]

    ## 调试使用
    #print(data_transformation_extratct[0:20])
    ##print(data_transformation_extratct.size)
    #print(data_transformation_extratct.shape)

    ##筛选 xyz 和两个角速度
    #data_transformation_extratct = data_transformation_extratct[ : , [1,2,6]]

    ## 调试使用
    #print(data_transformation_extratct[0:20])
    ##print(data_transformation_extratct.size)
    #print(data_transformation_extratct.shape)


    # ----------------2.速度信息--------------

    # 将数据转化成载入并转化成numpy
    data_odom = np.loadtxt('odom.txt',dtype= 'str',skiprows=0,delimiter=",")


    #只筛选特定的行
    #data_odom_extratct = data_odom[1:20, [0,48,49,50,51,52,53]]
    data_odom_extratct = data_odom[ : , [0,5,6,7,8,9,10,11,48,49,50,51,52,53]]

    # 针对平面问题的数据筛选

    #只筛选特定的行
    data_odom_extratct = data_odom_extratct[ : , [1,2,6,8,9,13]]


    ### ----------------3.输入信息--------------

    ### 将数据转化成载入并转化成numpy
    ##data_cmd_vel = np.loadtxt('cmd_vel.txt',dtype= 'str',skiprows=0,delimiter=",")

    ###筛选特定的行
    ###data_cmd_vel = data_cmd_vel[1:20,:]
    ##data_cmd_vel = data_cmd_vel[:,:]

    ### 调试使用
    ###print(data_cmd_vel)
    ##print(data_cmd_vel.size)
    ##print(data_cmd_vel.shape)

    ## 可以不用筛选特定的行

    ### ----------------4.时间信息--------------

    ### 将数据转化成载入并转化成numpy
    ##data_clock = np.loadtxt('clock.txt',dtype= 'str',skiprows=0,delimiter=",")

    ###筛选特定的行
    ###data_clock = data_clock[1:20,:]
    ##data_clock = data_clock[:,:]

    ### 调试使用
    ###print(data_clock)
    ##print(data_clock.size)
    ##print(data_clock.shape)

    ### 可以不用筛选特定的行


    # --------------------将数据拼接在一个numpy当中---------------------

    # 取数据前2000行
    data_odom_extratct_data_size = data_odom_extratct[ 1:data_size+1 , :]

    print("data_odom_extratct_data_size")
    print(data_odom_extratct_data_size[0:20,:])



    ## ----------------去数据前2000行--------------

    #data_transformation_2000 = data_transformation_extratct[ 1:data_size+1 , 1:8] # 第零行是标签，取两千个数据 

    #print("data_transformation_2000")
    ##print(data_transformation_2000[0:5,:])
    #print(data_transformation_2000.shape)

    #data_odom_2000 = data_odom_extratct[ 1:data_size+1 , 1:7] # 第零行是标签，取两千个数据

    #print("data_odom_2000.size")
    ##print(data_odom_2000[0:5,:])
    #print(data_odom_2000.shape)

    #生成输入数据 
    data_input = data_odom_extratct_data_size.astype(np.float)

    print("data_input")
    print(data_input[1:20,:])

    # 定义乘数
    data_input[:, 2] = data_input[:, 2] *10
    data_input[:, 3] = data_input[:, 3] *1000
    data_input[:, 4] = data_input[:, 4] *1000
    data_input[:, 5] = data_input[:, 5] *1000

    data_input = 10*data_input

    # 生成输出数据 
    data_output = data_odom_extratct[12:data_size+12 ,[0,1,2] ].astype(np.float)

    # 定义乘数
    data_output[:, 2] = data_output[:, 2] *10

    data_output = 10*data_output

    ### ----------------生成指定cmd_vel数据--------------

    ##data_cmd_vel_2000 = np.zeros((data_size,6))

    ##data_cmd_vel_2000[:,0]=0.20
    ##data_cmd_vel_2000[:,5]=0.05

    ###print(data_cmd_vel_2000)

    ### 将数据拼接在一起

    ##data_input = np.append(data_transformation_2000,data_odom_2000, axis=1).astype(np.float)

    ##data_input = np.append(data_input,data_cmd_vel_2000, axis=1)# tensor避免数据损失

    ##print("data_input")
    ##print(data_input[data_size-6:data_size,:])
    ##print(data_input.shape)

    ### ----------------生成输出数据--------------

    ##data_output = data_transformation_extratct[ 2:data_size+2 , 1:8].astype(np.float) # tensor避免数据损失

    ##print("data_output")
    ##print(data_output[data_size-6:data_size,:]) 


    #### ----------------数据预处理：归一化--------------

    ##data_input -= np.mean(data_input, axis = 0) # zero-center
    ##data_output -= np.mean(data_output, axis = 0) # zero-center

    ##data_input /= np.std(data_input, axis = 0) # normalize
    ##data_output /= np.std(data_output, axis = 0) # normalize

    ###print("data_input预处理后")
    ###print(data_input[data_size-6:data_size,:])
    ###print(data_input.shape)

    ###print("data_output预处理后")
    ###print(data_output[data_size-6:data_size,:]) 


    # ------------将numpy转化成tensor-----------

    data_input_torch = torch.from_numpy(data_input)
    data_input_float = data_input_torch.float() # 转化成浮点数

    data_output_torch = torch.from_numpy(data_output)
    data_output_float = data_output_torch.float() # 转化成浮点数 

    print("data_input_torch")
    print(data_input_torch[1000:1200,:]) 
    print(data_input_torch.shape)


    print("data_output_torch")
    print(data_output_torch[1000:1200,:]) 
    print(data_output_torch.shape)

    # ------------进行批训练-----------

    BATCH_SIZE = 30

    # 定义数据库 （输入输出分别是之前的输入输出）
    dataset = Data.TensorDataset(data_input_float, data_output_float) 

    # 定义数据加载器
    loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)




    #----------------定义相关网络-----------------

    # 定义迭代次数
    times = data_size

    # 生成随机输出变量
    data_plot = torch.zeros(times,3) #设定了一千条数据

    # 生成损失函数误差变量
    loss_plot = torch.zeros(times,3) #设定了一千条数据

    # 首先，定义所有层属性
    class Net(torch.nn.Module):  # 继承 torch 的 Module
    
        #定义该神经网络：4个全连接层，每层元素128个
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.fc1 = torch.nn.Linear(n_feature, n_hidden)   # 第一个全连接层
            self.fc2 = torch.nn.Linear(n_hidden, n_hidden)   # 第二个全连接层
            self.fc3 = torch.nn.Linear(n_hidden, n_hidden)   # 第三个全连接层
            self.fc4 = torch.nn.Linear(n_hidden, n_output)   # 第四个全连接层
            self.dropout = torch.nn.Dropout(p=0.35)
    
        #定义前向网络
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            x = self.dropout(x)
            return x


    net = Net(n_feature=6, n_hidden=9, n_output=3) 

    print(net)


    #----------------定义优化方法&定义损失函数-----------------

    #使用“随机梯度下降法”进行参数优化
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率

    #使用“ADAM”进行参数优化
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003) # 传入 net 的所有参数, 学习率

    #定义损失函数，计算均方差
    #loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
    loss_func = torch.nn.L1Loss()      # 预测值和真实值的误差计算公式 (均方差)

    #----------------使用cuda进行GPU计算-----------------

    net.cuda()
    loss_func.cuda()

    #----------------具体训练过程-----------------

    for epoch in range(2):
        for step, (batch_x, batch_y) in enumerate(loader):

            prediction = net( batch_x.cuda() )     # input x and predict based on x

            loss = 1000*loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)

            print ("loss")
            print (loss)

            #loss_plot[t,:] = loss # 将损失函数赋值给绘画变量

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            print ('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| betch y: ', batch_y.numpy())

            ##计算误差百分数,并储存在data_plot当中
            #if t % 5 == 0:
            #    percent = 100*(prediction - batch_y.cuda())/batch_y.cuda()

            #    print ("percent")
            #    print (percent)

            #计算误差百分数,并储存在data_plot当中
           
            percent = 100*(prediction - batch_y.cuda())/batch_y.cuda()

            print ("percent")
            print (percent)

                #data_plot[t,:] = percent

                #print("prediction")
                #print(prediction)

            #print("\n\nprediction")
            #print(prediction)

            #print("\nactually")
            #print(y[:,t])

            #print("\nprediction - actually")
            #print(prediction - y[:,t])

            #print("\npercent")
            #print(percent)

    ##计算3个输出的平均值（把每列加起来）

    ##将数据从tensor转化成numpy
    #data_plot_numpy = data_plot.detach().numpy()

    ##print("data_plot_numpy")
    ##print(data_plot_numpy[1:20,:])
    ##print(data_plot_numpy.shape)

    ##print("data_plot_numpy转置")
    ##print(data_plot_numpy.T[:,1:20])
    ##print(data_plot_numpy.T.shape)


    ##取每个元素的绝对值
    #data_plot_numpy_abs = np.abs(data_plot_numpy.T) #需要进行转置才能得到相关数据

    ##调用sum函数，将每列数据加起来，求误差的平均数
    #average = np.sum(data_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

    ##print("\naverage")
    ##print(average)


    ##将误差平均值可视化
    #X = np.linspace(1, times, times, endpoint=True)

    ##plt.xlim(times-200, times)#设置XY轴的显示范围
    ##plt.ylim(0, 100)

    #plt.plot(X,average)
    #plt.show()

    ###将X、Y、Z误差可视化
    ##for num in range (0,7):
    ##    X = np.linspace(1, times, times, endpoint=True)
    ##    plt.plot(X,data_plot[:,num].detach().numpy())

    ##    #plt.xlim(1, times)#设置XY轴的显示范围
    ##    #plt.ylim(0, 100)

    ##    plt.show()

    ##----------------将loss函数可视化-----------------


    ##将数据从tensor转化成numpy
    #loss_plot_numpy = loss_plot.detach().numpy()

    #print("data_plot_numpy")
    #print(loss_plot_numpy[1:20,:])
    #print(loss_plot_numpy.shape)

    #print("loss_plot_numpy转置")
    #print(loss_plot_numpy.T[:,1:200])
    #print(loss_plot_numpy.T.shape)

    ##取每个元素的绝对值
    #loss_plot_numpy_abs = np.abs(loss_plot_numpy.T) #需要进行转置才能得到相关数据

    ##调用sum函数，将每列数据加起来，求误差的平均数
    #average = np.sum(loss_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均


    ##将误差平均值可视化
    #X = np.linspace(1, times, times, endpoint=True)

    ##plt.xlim(times-200, times)#设置XY轴的显示范围
    ##plt.ylim(0, 50)

    #plt.plot(X,average)
    #plt.show()

